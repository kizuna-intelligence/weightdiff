import json
from dataclasses import replace
from pathlib import Path

import gguf
import numpy as np
import pytest
from safetensors.numpy import save_file

from weightdiff.application.collect import build_report, snapshot
from weightdiff.application.report import metric_cards, to_markdown
from weightdiff.domain.quant import classify_quant_samples, compare_quant_sample, plan_quant_sample
from weightdiff.domain.model import SampleResult
from weightdiff.external import gguf as headers, gguf_names, hub


@pytest.fixture
def quant_models(tmp_path):
    def make(qtype='Q8_0', changed=None, sign=False, layers=4, shape=(16, 256)):
        rng = np.random.default_rng(42)
        a = tmp_path / 'hf'
        a.mkdir(exist_ok=True)
        tensors = {}
        b = tmp_path / f'{qtype}.gguf'
        writer = gguf.GGUFWriter(b, 'llama')
        writer.add_block_count(layers)
        writer.add_embedding_length(shape[-1])
        writer.add_chat_template('test template')
        for i in range(layers):
            for kind, gg in [('q_proj', 'attn_q'), ('k_proj', 'attn_k'), ('v_proj', 'attn_v'), ('o_proj', 'attn_output')]:
                name = f'model.layers.{i}.self_attn.{kind}.weight'
                values = rng.normal(size=shape).astype(np.float16)
                tensors[name] = values
                other = values.astype(np.float32)
                if sign and kind == 'q_proj':
                    other = -other
                packed = gguf.quants.quantize(other, gguf.GGMLQuantizationType[qtype])
                if changed == (i, kind):
                    # Change exactly one encoded block, preserving a valid GGUF tensor.
                    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[qtype]]
                    altered = gguf.quants.quantize(np.full(block, 50, dtype=np.float32), gguf.GGMLQuantizationType[qtype])
                    packed.reshape(-1)[:size] = altered.reshape(-1)
                writer.add_tensor(f'blk.{i}.{gg}.weight', packed, raw_dtype=gguf.GGMLQuantizationType[qtype])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        save_file(tensors, str(a / 'model.safetensors'))
        (a / 'config.json').write_text(json.dumps({'model_type':'llama','num_hidden_layers':layers,'hidden_size':shape[-1]}))
        (a / 'tokenizer_config.json').write_text(json.dumps({'chat_template':'test template'}))
        return str(a), str(b)
    return make


def test_f16_to_q8_requantizes_exactly(quant_models):
    a, b = quant_models(shape=(64, 256))
    report = build_report(a, b)
    assert len(report.samples) == 16
    assert {s.status for s in report.samples} == {'same_after_requant'}
    assert report.quant_noise[0].dtype == 'Q8_0'
    assert report.quant_noise[0].median > 0
    assert report.quant_noise[0].undetectable_percent == report.quant_noise[0].median * 200
    assert report.layer_summary[0].a_count == report.layer_summary[0].b_count == 4
    assert len(report.b.original_names) == 16
    assert not report.a.unmapped_names and not report.b.unmapped_names
    assert not any('safetensors が無い' in f.title for f in report.flags)
    assert 'Quantized values' in metric_cards(report)
    assert 'quantization' in to_markdown(report).lower()


def test_one_q8_block_changed_is_beyond_noise(quant_models):
    a, b = quant_models(changed=(2, 'v_proj'))
    report = build_report(a, b)
    beyond = [s for s in report.samples if s.status == 'beyond_quant_noise']
    assert len(beyond) == 1
    assert beyond[0].name == 'model.layers.2.self_attn.v_proj.weight'
    assert sum(s.status == 'same_after_requant' for s in report.samples) == 15


def test_q4_noise_without_requant_encoder(monkeypatch, quant_models):
    a, b = quant_models(qtype='Q4_0')
    # K encoders are not implemented in gguf-py; exercise the same fallback on Q4_0.
    def unavailable(*args):
        raise NotImplementedError('encoder unavailable')
    monkeypatch.setattr('weightdiff.domain.quant.quantize', unavailable)
    report = build_report(a, b)
    assert {s.status for s in report.samples} == {'within_quant_noise'}
    assert report.quant_noise[0].dtype == 'Q4_0'


def test_sign_flips_are_verified_as_transforms(quant_models):
    a, b = quant_models(sign=True)
    report = build_report(a, b)
    assert sum(s.status == 'same_after_transform' for s in report.samples) == 4
    assert {s.transform for s in report.samples if s.status == 'same_after_transform'} == {'−x'}
    assert sum(s.status == 'beyond_quant_noise' for s in report.samples) == 0


def test_systematic_requires_all_layers_even_when_subsampled(quant_models):
    a, b = quant_models(sign=True)
    report = build_report(a, b, max_tensors=2)
    assert not any(s.status == 'systematic' for s in report.samples)


@pytest.mark.parametrize('dtype', ['Q8_0','Q4_0','F16'])
def test_gguf_against_itself_uses_exact_block_bytes(quant_models, dtype):
    _, b = quant_models(qtype=dtype)
    report = build_report(b, b)
    assert all(s.status == 'same' for s in report.samples)
    assert report.layer_summary[0].a_count == 4
    assert report.b.original_names == {}


def test_two_gguf_dtypes_are_dequantized(quant_models):
    _, q8 = quant_models(qtype='Q8_0')
    _, q4 = quant_models(qtype='Q4_0')
    report = build_report(q8, q4)
    assert {s.status for s in report.samples} == {'within_quant_noise'}
    assert {n.dtype for n in report.quant_noise} == {'Q8_0','Q4_0'}


def test_reverse_float_and_gguf_reference(quant_models):
    a, b = quant_models()
    report = build_report(b, a)
    assert {s.status for s in report.samples} == {'within_quant_noise'}
    assert len(report.a.original_names) == 16


def test_progressive_header_reading(quant_models, monkeypatch):
    _, path = quant_models()
    data = Path(path).read_bytes()
    monkeypatch.setattr(headers, 'INITIAL_HEADER', 32)
    calls = []
    def read(length):
        calls.append(length)
        return data[:length]
    parsed = headers.read_header(read, len(data))
    assert parsed['tensors']['blk.0.attn_q.weight']['shape'] == (16,256)
    assert len(calls) > 3 and all(b == 2*a for a,b in zip(calls,calls[1:]))
    assert parsed['tensors']['blk.0.attn_q.weight']['data_start'] % 32 == 0
    assert parsed['header_bytes'] <= calls[-1] < len(data)


def test_header_limit_and_truncation(quant_models, monkeypatch):
    _, path = quant_models()
    data = Path(path).read_bytes()
    monkeypatch.setattr(headers,'INITIAL_HEADER',32)
    monkeypatch.setattr(headers,'MAX_HEADER',128)
    with pytest.raises(ValueError, match='exceed'):
        headers.read_header(lambda n:data[:n])
    monkeypatch.setattr(headers,'MAX_HEADER',256 * 1024 * 1024)
    with pytest.raises(ValueError, match='Truncated'):
        headers.read_header(lambda n:data[:n], 32)
    with pytest.raises(ValueError, match='magic'):
        headers.parse_header(b'NOPE')


def test_metadata_config_and_name_mapping():
    metadata={'general.architecture':'qwen35','qwen35.block_count':32,'qwen35.embedding_length':2560,
              'tokenizer.ggml.tokens':['a','b']}
    assert headers.config_from_gguf(metadata) == {'model_type':'qwen35','num_hidden_layers':32,'hidden_size':2560,'vocab_size':2}
    names=['model.language_model.layers.0.self_attn.q_proj.weight','model.language_model.layers.0.linear_attn.A_log',
           'model.language_model.norm.weight','model.language_model.layers.0.self_attn.q_proj.bias','unknown']
    available=['blk.0.attn_q.weight','blk.0.ssm_a','output_norm.weight','blk.0.attn_q.bias']
    mapped=gguf_names.map_names(names,available,'qwen35',32)
    assert mapped == dict(zip(names[:4],available))
    assert gguf_names.map_names(names,available,'unknown',32) == {}


def test_unmapped_names_are_counted_and_preserved(quant_models):
    a,b=quant_models()
    # An unsupported architecture must not invent correspondences.
    data=Path(b).read_bytes().replace(b'llama',b'other')
    Path(b).write_bytes(data)
    report=build_report(a,b)
    assert len(report.a.unmapped_names)==16 and len(report.b.unmapped_names)==16
    assert report.samples == ()
    assert 'No correspondence' in to_markdown(report)
    assert len(report.tensor_diff.only_a) == len(report.tensor_diff.only_b) == 16


def test_block_window_alignment(quant_models):
    a,b=quant_models(shape=(64,256))
    sa,sb=snapshot(a),snapshot(b)
    ta=next(iter(sa.tensors.values()));tb=sb.tensors['blk.0.attn_q.weight']
    # Explicit Q4_K layout: 256 elements / 144 bytes.
    tb=replace(tb,dtype='Q4_K',nbytes=64*144)
    plan=plan_quant_sample(ta,tb,ta.name,sa.format,sb.format)
    assert plan.skip_reason is None
    assert all((start-tb.data_start)%144==0 and length%144==0 for start,length in plan.b_ranges)
    assert [length for _,length in plan.b_ranges]==[1152,1152]


def test_unsupported_quant_type_is_skipped_with_dtype(quant_models,monkeypatch):
    a,b=quant_models()
    sa,sb=snapshot(a),snapshot(b)
    ta=next(iter(sa.tensors.values()));tb=sb.tensors['blk.0.attn_q.weight']
    plan=plan_quant_sample(ta,tb,ta.name,sa.format,sb.format)
    from weightdiff.domain import quant
    original=quant.dequantize
    def read(data,typ):
        if typ==gguf.GGMLQuantizationType.Q8_0:raise NotImplementedError('not supported')
        return original(data,typ)
    monkeypatch.setattr(quant,'dequantize',read)
    from weightdiff.external import local
    chunks=[local.read_ranges(str(Path(b).parent),None,Path(b).name,plan.b_ranges)]
    result=compare_quant_sample(plan,[b'\0'*n for _,n in plan.a_ranges],chunks[0])
    assert result.status=='skipped' and 'Q8_0' in result.reason


def test_split_gguf_reads_all_shards_and_rejects_missing(tmp_path):
    paths=[]
    for i in range(2):
        path=tmp_path / f'tiny-{i+1:05d}-of-00002.gguf'
        paths.append(path)
        writer=gguf.GGUFWriter(path,'llama')
        writer.add_block_count(2)
        writer.add_tensor(f'blk.{i}.attn_norm.weight',np.ones(32,dtype=np.float32))
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    snap=snapshot(str(paths[1]))
    assert len(snap.tensors)==2 and len(snap.files)==2
    report=build_report(str(paths[0]),str(paths[1]))
    assert len(report.samples)==2 and all(s.status=='same' for s in report.samples)
    paths[0].unlink()
    with pytest.raises(ValueError,match='Incomplete split'):
        snapshot(str(paths[1]))


def test_remote_gguf_uses_bounded_http_ranges(quant_models,monkeypatch):
    _,path=quant_models()
    payload=Path(path).read_bytes()
    filename='tiny.gguf'
    monkeypatch.setattr(hub,'resolve_revision',lambda *args:'pinned-sha')
    monkeypatch.setattr(hub,'list_files',lambda *args:[filename])
    monkeypatch.setattr(hub,'file_size',lambda *args:len(payload))
    calls=[]
    def read(repo,revision,name,start,length,category):
        assert (repo,revision,name)==('owner/repo','pinned-sha',filename)
        calls.append((start,length,category))
        return payload[start:start+length]
    monkeypatch.setattr(hub,'_range',read)
    monkeypatch.setattr(headers,'INITIAL_HEADER',128)
    report=build_report('owner/repo:tiny.gguf','owner/repo:tiny.gguf',max_tensors=1)
    assert report.a.revision=='pinned-sha'
    assert report.samples[0].status=='same'
    assert any(c[2]=='header_bytes' for c in calls) and any(c[2]=='sample_bytes' for c in calls)
    assert max(c[1] for c in calls)<len(payload)


def test_same_typed_gguf_changed_block_is_different(quant_models,tmp_path):
    _,original=quant_models()
    saved=tmp_path/'original.gguf'
    saved.write_bytes(Path(original).read_bytes())
    _,changed=quant_models(changed=(0,'q_proj'))
    report=build_report(str(saved),changed)
    assert sum(s.status=='different' for s in report.samples)==1
    assert sum(s.status=='same' for s in report.samples)==15


def test_truncated_sample_is_skipped(quant_models):
    a,b=quant_models()
    sa,sb=snapshot(a),snapshot(b)
    ta=next(iter(sa.tensors.values()));tb=sb.tensors['blk.0.attn_q.weight']
    plan=plan_quant_sample(ta,tb,ta.name,sa.format,sb.format)
    sample=compare_quant_sample(plan,[b''],[b''])
    assert sample.status=='skipped' and 'incomplete' in sample.reason


def test_systematic_uses_sampled_layers_even_if_another_is_unsampled():
    def value(i,error,status='pending_quant'):
        return SampleResult(f'model.layers.{i}.x.weight',status,'',relative_error=error,quant_types=('Q8_0',))
    samples=[value(i,.005,'same_after_requant') for i in range(10)]
    samples += [value(i,2) for i in range(3)]
    # Distinguish control names so they do not claim duplicate layer evidence.
    samples[:10]=[replace(s,name=f'control{i}') for i,s in enumerate(samples[:10])]
    classified,_=classify_quant_samples(samples,{'model.layers.{i}.x.weight':{0,1,2,3}})
    assert sum(s.status=='systematic' for s in classified)==3
    # A sampled skipped/within layer, unlike an unsampled one, blocks grouping.
    classified,_=classify_quant_samples(samples+[value(3, .005, 'skipped')])
    assert sum(s.status=='beyond_quant_noise' for s in classified)==3



def test_medium_quantized_tensor_also_has_center_window(quant_models):
    a,b=quant_models(shape=(16,256))
    sa,sb=snapshot(a),snapshot(b)
    ta=next(iter(sa.tensors.values()));tb=sb.tensors['blk.0.attn_q.weight']
    plan=plan_quant_sample(ta,tb,ta.name,sa.format,sb.format)
    assert len(plan.a_ranges)==len(plan.b_ranges)==2
    assert plan.a_ranges[1]==(ta.data_start+1024*2,2048*2)
    assert plan.b_ranges[1]==(tb.data_start+1024//32*34,2048//32*34)



def test_systematic_can_span_mixed_quantization_types():
    samples=[]
    for dtype in ('Q4_K','Q6_K'):
        samples.extend(SampleResult(f'control_{dtype}_{i}','within_quant_noise','',relative_error=.01,quant_types=(dtype,))
                       for i in range(10))
    samples.extend(SampleResult(f'model.layers.{i}.x.weight','pending_quant','',relative_error=2,
                               quant_types=(('Q4_K' if i%2 else 'Q6_K'),)) for i in range(4))
    classified,_=classify_quant_samples(samples,{'model.layers.{i}.x.weight':{0,1,2,3}})
    assert sum(s.status=='systematic' for s in classified)==4
    assert sum(s.status=='beyond_quant_noise' for s in classified)==0
