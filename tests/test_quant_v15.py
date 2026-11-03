"""§15 regression cases use only small generated local tensors and GGUF files."""
import json
from dataclasses import replace

import gguf
import numpy as np
import pytest
from safetensors.numpy import save_file

from weightdiff.application.collect import build_report
from weightdiff.application.report import metric_cards, quantization_markdown, samples_markdown
from weightdiff.domain.model import SampleResult
from weightdiff.domain.quant import classify_quant_samples
from weightdiff.external import ggml, local


@pytest.mark.parametrize(('transform', 'label'), [
    (lambda a, i: -np.exp(a), '−exp(x)'),
    (lambda a, i: a + 1, 'x+1'),
    (lambda a, i: a - 1, 'x−1'),
    (lambda a, i: np.exp(a), 'exp(x)'),
])
def test_element_transforms_are_verified(model_pair, transform, label):
    report = build_report(*model_pair(transform, dtype='F32'))
    assert report.samples[0].status == 'same_after_transform'
    assert report.samples[0].transform == label
    assert report.samples[0].threshold == 1e-5
    assert report.samples[0].expected_errors == (('F32', 0.0),)
    assert '変換' in report.samples[0].reason
    assert 'beyond 0 · uniform 0' in metric_cards(report)
    assert label in samples_markdown(report)


def test_log_checks_only_positive_original_values(model_pair):
    def transform(a, i):
        return np.log(np.abs(a))
    report = build_report(*model_pair(transform, dtype='F32'))
    assert report.samples[0].status == 'same_after_transform'
    assert report.samples[0].transform == 'log(x) (x>0)'


@pytest.mark.parametrize(('transform', 'kind', 'dtype'), [
    (lambda a, i: np.roll(a, 128, axis=1), 'Column', 'F32'),
    (lambda a, i: a[::-1], 'Row', 'Q8_0'),
])
def test_rearrangement_checks_complete_rows_and_bounded_full_a(model_pair, monkeypatch, transform, kind, dtype):
    # The same head-group permutation must match all three complete rows.
    a, b = model_pair(transform, dtype=dtype)
    calls = []
    read = local.read_ranges
    def tracked(repo, revision, filename, ranges):
        calls.append((filename, ranges))
        return read(repo, revision, filename, ranges)
    monkeypatch.setattr(local, 'read_ranges', tracked)
    report = build_report(a, b)
    result = report.samples[0]
    assert result.status == 'rearranged' and kind in result.transform
    assert '並べ替えると同じ' in result.reason
    assert bool([r for f, r in calls if f.endswith('.safetensors') and r[0][1] == 16 * 256 * 4]) == (kind == 'Row')
    assert all(not (f.endswith('.gguf') and any(n == 16 * 256 // 32 * 34 for _, n in r)) for f, r in calls)


def test_uniform_three_percent_shift_is_above_a_noise_floor(model_pair):
    report = build_report(*model_pair(lambda a, i: a * 1.03, count=1))
    sample = report.samples[0]
    assert sample.status == 'beyond_quant_noise'
    assert .029 < sample.relative_error < .032
    assert sample.expected_errors[0][1] < .01
    assert sample.relative_error > sample.threshold
    assert report.quant_noise[0].source == 'A'
    assert report.quant_noise[0].undetectable_percent < 2


def test_three_percent_shift_on_all_sampled_values_cannot_raise_the_floor(model_pair):
    # One sampled tensor also exercises the case where B's median is its entire error.
    report = build_report(*model_pair(lambda a, i: a * 1.03))
    expected = report.samples[0].expected_errors[0][1]
    assert report.quant_noise[0].median == expected
    assert report.quant_noise[0].median < report.samples[0].relative_error / 2
    assert 'observed error in B' not in quantization_markdown(report)


def test_k_type_without_llama_cpp_falls_back_with_visible_warning(tmp_path, monkeypatch):
    folder = tmp_path / 'hf'
    folder.mkdir()
    values = np.ones((4, 256), dtype=np.float32)
    save_file({'model.layers.0.self_attn.q_proj.weight': values}, str(folder / 'model.safetensors'))
    (folder / 'config.json').write_text('{"model_type":"llama","num_hidden_layers":1}')
    target = tmp_path / 'q4.gguf'
    writer = gguf.GGUFWriter(target, 'llama')
    writer.add_block_count(1)
    # Valid zero Q4_K blocks; deliberately independent of the optional encoder.
    writer.add_tensor('blk.0.attn_q.weight', np.zeros((4, 144), dtype=np.uint8),
                      raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    def absent():
        raise ggml.Unavailable('llama-cpp-python is not installed')
    monkeypatch.setattr(ggml, '_library', absent)
    report = build_report(str(folder), str(target))
    assert report.samples[0].status == 'within_quant_noise'
    assert report.quant_noise[0].source == 'B'
    assert 'llama-cpp-python is not installed' in report.quant_noise[0].detail
    assert '全体を一様に変えた学習は見分けられません' in quantization_markdown(report)


@pytest.mark.parametrize('dtype', ['Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'IQ4_XS'])
def test_optional_native_roundtrip_matches_gguf_py_dequantization(dtype):
    values = np.random.default_rng(10).normal(size=2048).astype(np.float32)
    native = ggml.roundtrip(values, dtype)
    if 'error' in native and 'not installed' in native['error']:
        pytest.skip('optional llama-cpp-python is not installed')
    assert 'error' not in native, native
    decoded = gguf.quants.dequantize(np.frombuffer(native['bytes'], dtype=np.uint8),
                                    gguf.GGMLQuantizationType[dtype])
    np.testing.assert_allclose(native['values'], decoded, rtol=1e-6, atol=1e-6)


def test_k_type_reference_floor_with_optional_native_encoder(tmp_path):
    values = np.random.default_rng(3).normal(size=(8, 256)).astype(np.float32)
    native = ggml.roundtrip(values, 'Q4_K')
    if 'error' in native and 'not installed' in native['error']:
        pytest.skip('optional llama-cpp-python is not installed')
    assert 'error' not in native
    folder = tmp_path / 'hf'
    folder.mkdir()
    save_file({'model.layers.0.self_attn.q_proj.weight': values}, str(folder / 'model.safetensors'))
    (folder / 'config.json').write_text('{"model_type":"llama","num_hidden_layers":1}')
    target = tmp_path / 'native.gguf'
    writer = gguf.GGUFWriter(target, 'llama'); writer.add_block_count(1)
    writer.add_tensor('blk.0.attn_q.weight', np.frombuffer(native['bytes'],dtype=np.uint8).reshape(8,144),
                      raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    report = build_report(str(folder), str(target))
    assert report.samples[0].status == 'same_after_requant'
    assert report.quant_noise[0].source == 'A'
    assert 'B (fallback)' not in quantization_markdown(report)


def test_systematic_requires_three_sampled_layers_and_all_beyond():
    def sample(i, status='pending_quant', error=.03):
        return SampleResult(f'model.layers.{i}.x.weight', status, '', relative_error=error,
                            quant_types=('Q8_0',), expected_errors=(('Q8_0', .006),))
    samples = [sample(i) for i in [0, 4, 8]]
    result, _ = classify_quant_samples(samples, {'model.layers.{i}.x.weight': set(range(12))})
    assert {s.status for s in result} == {'systematic'}
    result, _ = classify_quant_samples(samples[:2])
    assert {s.status for s in result} == {'beyond_quant_noise'}
    result, _ = classify_quant_samples(samples + [sample(9, 'skipped')])
    assert sum(s.status == 'beyond_quant_noise' for s in result) == 3
    result, _ = classify_quant_samples(samples + [sample(9, error=.005)])
    assert sum(s.status == 'beyond_quant_noise' for s in result) == 3
    assert result[-1].status == 'within_quant_noise'


def test_missing_k_sample_data_stays_skipped():
    from weightdiff.domain.model import TensorInfo
    from weightdiff.domain.quant import native_calibration_inputs, plan_quant_sample, compare_quant_sample
    ta = TensorInfo('x', 'F32', (256,), 1024, 'a', 0)
    tb = TensorInfo('x', 'Q4_K', (256,), 144, 'b', 0)
    plan = plan_quant_sample(ta, tb, 'x', 'safetensors', 'gguf')
    assert native_calibration_inputs(plan, [b'\0' * 1024], 'connection failed') == {}
    assert native_calibration_inputs(plan, [b'\0' * 1024], [b'']) == {}
    result = compare_quant_sample(plan, [b'\0' * 1024], 'connection failed')
    assert result.status == 'skipped' and 'connection failed' in result.reason


def test_imatrix_required_types_fall_back_without_calling_native_quantizer(monkeypatch):
    class Lib:
        def ggml_type_name(self, qtype): return b'iq2_xxs'
        def ggml_quantize_requires_imatrix(self, qtype): return True
        def ggml_quantize_chunk(self, *args): raise AssertionError('imatrix must not be fabricated')
    monkeypatch.setattr(ggml, '_library', lambda: Lib())
    result = ggml.roundtrip(np.ones(256, dtype=np.float32), 'IQ2_XXS')
    assert 'requires an importance matrix' in result['error']
