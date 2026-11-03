import json
from dataclasses import replace

import numpy as np
import pytest
from safetensors import TensorSpec, serialize
from safetensors.numpy import save_file

from weightdiff.application.collect import build_report
from weightdiff.application.report import layer_rows, metric_cards, to_markdown, to_json
from weightdiff.domain.compare import region_stats, plan_regions, diff_tensors
from weightdiff.domain.flags import make_flags
from weightdiff.domain.layers import axis_map
from weightdiff.domain.model import ModelSnapshot, TensorInfo, RegionPlan, RegionStats


@pytest.fixture
def widened_pair(tmp_path):
    def make(position=0, insertions=1, mixed_dtype=False):
        a, b = tmp_path / 'a', tmp_path / 'b'
        a.mkdir(); b.mkdir()
        rng = np.random.default_rng(18)
        original = {'model.embed_tokens.weight': rng.normal(0,.3,(1024,8)).astype('float32'),
                    'model.norm.weight': rng.uniform(-.5,.5,8).astype('float32')}
        for i, kind in enumerate(['linear','linear','linear','full']):
            original[f'model.layers.{i}.{kind}.weight'] = rng.normal(0,.3,(256,8)).astype('float32')
            original[f'model.layers.{i}.input_layernorm.weight'] = rng.uniform(-.5,.5,8).astype('float32')
        wider = {}
        c = np.sqrt(8/10)
        for name, value in original.items():
            target = name
            if name.startswith('model.layers.'):
                parts = name.split('.'); i=int(parts[2]); parts[2]=str(i+insertions if i>=position else i)
                target='.'.join(parts)
            shape = tuple(10 if v==8 else (1025 if v==1024 else v) for v in value.shape)
            out = np.zeros(shape,dtype='float32')
            if value.ndim==1: out[:8] = c*(1+value)-1
            else: out[:value.shape[0],:8]=value
            wider[target]=out
        for j in range(insertions):
            out=np.zeros((256,10),dtype='float32');out[0,0]=1e12
            wider[f'model.layers.{position+j}.linear.weight']=out
            wider[f'model.layers.{position+j}.input_layernorm.weight']=np.zeros(10,dtype='float32')
        wider['model.embed_tokens.weight'][:1024,8]=1e12
        head=np.zeros((1025,10),dtype='float32');head[:1024,:8]=original['model.embed_tokens.weight']
        head[1024,0]=1048576;head[0,9]=1048576
        wider['lm_head.weight']=head
        for path, tensors, width, tied in [(a,original,8,True),(b,wider,10,False)]:
            if path == b and mixed_dtype:
                encoded = {}
                buffers = []
                for name, value in tensors.items():
                    # Reproduce F32 -> BF16 within the repeated linear layers,
                    # including the matrix used to locate the insertion.
                    bf16 = '.layers.' in name and ('.linear.' in name or
                        ('.input_layernorm.' in name and int(name.split('.')[2]) < 3 + insertions))
                    data = (value.view('uint32') >> 16).astype('<u2') if bf16 else value
                    buffers.append(data)
                    encoded[name] = TensorSpec(dtype='bfloat16' if bf16 else 'float32',
                        shape=value.shape, data_ptr=data.ctypes.data, data_len=data.nbytes)
                (path/'model.safetensors').write_bytes(bytes(serialize(encoded)))
            else:
                save_file(tensors,str(path/'model.safetensors'))
            (path/'config.json').write_text(json.dumps({'hidden_size':width,'tie_word_embeddings':tied}))
        (a/'tokenizer_config.json').write_text(json.dumps({'chat_template':'same template'}))
        (b/'tokenizer_config.json').write_text('{}')
        (b/'chat_template.jinja').write_text('same template')
        return str(a),str(b)
    return make


@pytest.mark.parametrize('position',[0,2])
def test_widened_insertion_position_and_new_values(widened_pair,position):
    a,b=widened_pair(position)
    report=build_report(a,b,max_tensors=160)
    stack,=report.layer_summary
    assert stack.pairing_basis=='width'
    assert stack.added==(position,)
    assert stack.position_known and stack.position_basis=='samples'
    assert len(stack.equal_pairs)==4
    matrices=[s for s in report.samples if s.layer_pair and s.name.endswith(('linear.weight','full.weight'))]
    assert len(matrices)==4 and all(s.status=='embedded_top_left' for s in matrices)
    norms=[s for s in report.samples if s.name.endswith('layernorm.weight')]
    assert len(norms)==4 and all(s.status=='rescaled' for s in norms)
    assert all('幅 8→10' in s.reason and s.scale==pytest.approx(np.sqrt(.8),abs=1e-6) for s in norms)
    head,=[s for s in report.samples if s.name.endswith('→ lm_head.weight')]
    assert head.status=='embedded_top_left'
    assert any('ありえない大きさ' in f.title for f in report.flags)
    assert any('ほとんど 0' in f.title for f in report.flags)
    assert any(f.level=='info' and '置き場所' in f.title for f in report.flags)
    assert not any(f.level=='notice' and '会話' in f.title for f in report.flags)
    assert {r.region for r in report.new_regions}=={'extra_columns','extra_rows','extra_elements','new_tensor'}
    assert report.read_stats['position_bytes']>0 and report.read_stats['region_bytes']>0
    assert report.read_stats['total_bytes']==sum(v for k,v in report.read_stats.items() if k.endswith('_bytes') and k!='total_bytes')
    assert 'rescaled 5' in metric_cards(report)
    assert 'New regions in B' in to_markdown(report)
    assert json.loads(to_json(report))['new_regions']


def test_two_insertions_are_not_guessed(widened_pair):
    a,b=widened_pair(0,2)
    report=build_report(a,b)
    stack,=report.layer_summary
    assert stack.added_count==2 and not stack.position_known
    assert stack.position_basis=='structure' and stack.added==()
    assert report.read_stats['position_bytes']==0


@pytest.mark.parametrize('different',[False,True])
def test_effective_template_storage(model_factory,different):
    a=model_factory('a',tokenizer={'chat_template':'hello'})
    b=model_factory('b',template='changed' if different else 'hello')
    report=build_report(str(a),str(b))
    flags=[f for f in report.flags if '会話' in f.title]
    assert len(flags)==1
    assert flags[0].level==('notice' if different else 'info')


def test_transformers_legacy_processor_template_priority(model_factory):
    a=model_factory('a',template='ignored',tokenizer={'chat_template':'also ignored'})
    b=model_factory('b',template='different ignored')
    for root in (a,b): (root/'chat_template.json').write_text(json.dumps({'chat_template':'effective'}))
    report=build_report(str(a),str(b))
    assert not any(f.level=='notice' and '会話' in f.title for f in report.flags)


def test_training_like_vector_is_not_rescaled(tmp_path):
    rng=np.random.default_rng(10)
    values=rng.uniform(.1,1,2048).astype('float32')
    for name,data in [('a',values),('b',values*(1+rng.choice([-.02,.02],2048)).astype('float32'))]:
        folder=tmp_path/name;folder.mkdir()
        save_file({'norm.weight':data},str(folder/'model.safetensors'))
    result,=build_report(str(tmp_path/'a'),str(tmp_path/'b')).samples
    assert result.status=='different' and result.scale is None
    assert result.max_abs_a>0 and result.max_abs_b>0


def test_no_sample_reads_neither_positions_nor_regions(widened_pair):
    a,b=widened_pair()
    report=build_report(a,b,sample=False)
    assert not report.samples and not report.new_regions
    assert not report.layer_summary[0].position_known
    assert report.read_stats['sample_bytes']==report.read_stats['region_bytes']==report.read_stats['position_bytes']==0


def test_axis_map_drops_conflicts_and_existing_axes():
    def snap(shapes):
        return ModelSnapshot('',None,(),{},None,{n:TensorInfo(n,'F32',s,0,'',0) for n,s in shapes.items()},())
    a=snap({'one':(8,),'two':(8,),'three':(4,),'existing':(10,)})
    b=snap({'one':(12,),'two':(14,),'three':(10,)})
    assert axis_map(a,b)=={}


def test_new_region_nonfinite_and_truncation():
    plan=RegionPlan('w','new_tensor','f',((0,12),),'F32')
    stats=region_stats(plan,[np.array([0,1,np.inf],dtype='<f4').tobytes()])
    assert stats.count==3 and stats.zero_fraction==pytest.approx(1/3)
    assert not stats.finite and stats.max_abs==np.inf
    with pytest.raises(ValueError): region_stats(plan,[b''])


def test_region_plans_keep_priority_beyond_limit_and_exclude_gguf(widened_pair):
    a,b=widened_pair();report=build_report(a,b,sample=False)
    plans=plan_regions(report.a,report.b,report.tensor_diff,report.layer_summary,max_regions=3)
    assert len(plans)>3 and plans==sorted(plans,key=lambda p:(p.name,p.region))
    assert all(p.region=='new_tensor' or '.layers.' not in p.name for p in plans)
    assert plan_regions(report.a,report.b,report.tensor_diff,report.layer_summary,max_regions=0)==[]
    assert plan_regions(report.a,replace(report.b,format='gguf'),report.tensor_diff,report.layer_summary)==[]


def test_removed_repeated_layer_is_confirmed_by_values(tmp_path):
    rng=np.random.default_rng(20)
    layers=[rng.normal(0,.3,(128,8)).astype('float32') for _ in range(4)]
    for name,indices in [('a',[0,1,2,3]),('b',[0,2,3])]:
        root=tmp_path/name;root.mkdir()
        tensors={f'model.layers.{i}.matrix.weight':layers[j] for i,j in enumerate(indices)}
        save_file(tensors,str(root/'model.safetensors'))
    report=build_report(str(tmp_path/'a'),str(tmp_path/'b'))
    stack,=report.layer_summary
    assert stack.removed==(1,) and stack.position_known and stack.position_basis=='samples'
    assert stack.equal_pairs==((0,0),(2,1),(3,2))
    assert all(s.status=='same' for s in report.samples)


def test_two_removed_repeated_layers_remain_unknown(model_factory):
    a=model_factory('a',widths=(4,)*5)
    b=model_factory('b',widths=(4,)*3)
    stack,=build_report(str(a),str(b)).layer_summary
    assert stack.removed_count==2 and not stack.position_known
    assert stack.removed==() and stack.position_basis=='structure'


@pytest.mark.parametrize('position', [0, 2])
def test_width_alignment_ignores_dtype_and_locates_insertion(widened_pair, position):
    a, b = widened_pair(position, mixed_dtype=True)
    report = build_report(a, b)
    stack, = report.layer_summary
    assert stack.pairing_basis == 'width'
    assert stack.added == (position,)
    assert stack.position_known and stack.position_basis == 'samples'
    assert stack.equal_pairs == tuple((i, i + (i >= position)) for i in range(4))
    assert not stack.changed
    matrices = [s for s in report.samples if s.layer_pair and s.name.endswith(('linear.weight', 'full.weight'))]
    assert len(matrices) == 4 and all(s.status == 'embedded_top_left' for s in matrices)
    norms = [s for s in report.samples if s.name.endswith('layernorm.weight')]
    assert len(norms) == 4 and all(s.status == 'rescaled' for s in norms)
    assert all('幅 8→10' in s.reason for s in norms)
    head, = [s for s in report.samples if s.name.endswith('→ lm_head.weight')]
    assert head.status == 'embedded_top_left'
    new_layers = {int(r.name.split('.')[2]) for r in report.new_regions
                  if r.region == 'new_tensor' and '.layers.' in r.name}
    assert new_layers == {position}
    assert all('dtype ignored' in row[9] for row in layer_rows(report))
    sparse, = [f for f in report.flags if 'ほとんど 0' in f.title]
    assert f'B layer {position}' in sparse.detail and 'new_tensor' in sparse.detail
    extreme, = [f for f in report.flags if 'ありえない大きさ' in f.title]
    assert f'B layer {position}' in extreme.detail and 'new_tensor' in extreme.detail
    assert 'extra_columns' in extreme.detail and 'B outside layers' in extreme.detail


def test_mixed_dtype_multiple_insertions_remain_unknown(widened_pair):
    a, b = widened_pair(0, 2, mixed_dtype=True)
    report = build_report(a, b)
    stack, = report.layer_summary
    assert stack.pairing_basis == 'width' and stack.added_count == 2
    assert not stack.position_known and stack.position_basis == 'structure'
    assert stack.added == () and report.read_stats['position_bytes'] == 0
    assert not any('.layers.' in r.name and r.region == 'new_tensor' for r in report.new_regions)


def test_region_priority_and_tied_head_sampling(widened_pair):
    a, b = widened_pair(mixed_dtype=True)
    report = build_report(a, b)
    head = [r for r in report.new_regions if r.name == 'lm_head.weight']
    assert {r.region for r in head} == {'extra_columns', 'extra_rows'}
    assert next(r for r in head if r.region == 'extra_rows').count == 10
    sparse = [r for r in report.new_regions if r.region == 'new_tensor'
              and len(report.b.tensors[r.name].shape) == 2 and r.count >= 1024
              and r.zero_fraction >= .99]
    assert sparse and all(r.name.startswith('model.layers.0.') for r in sparse)
    flags = [f for f in report.flags if '広げた部分' in f.title]
    assert len(flags) == 2 and all(f.level == 'info' for f in flags)
    used, = [f for f in flags if '使われている' in f.title]
    assert 'embed_tokens.weight' in used.detail and 'lm_head.weight' in used.detail
    assert 'Widened regions are in use (' in to_markdown(report)


@pytest.mark.parametrize('new_count', [1, 161])
def test_priority_is_never_thinned_and_layer_extras_use_remaining_slots(new_count):
    def snapshot(shapes):
        return ModelSnapshot('', None, (), {}, None,
            {n: TensorInfo(n, 'F32', s, int(np.prod(s))*4, 'f', 0)
             for n, s in shapes.items()}, ())
    layer_names = [f'model.layers.0.matrix{i:03}.weight' for i in range(200)]
    a = snapshot({**{n: (2, 8) for n in layer_names}, 'model.embed_tokens.weight': (2, 8)})
    b = snapshot({**{n: (2, 10) for n in layer_names}, 'model.embed_tokens.weight': (2, 10),
                  **{f'new{i:03}.weight': (32, 32) for i in range(new_count)}})
    plans = plan_regions(a, b, diff_tensors(a, b, []), [])
    priority = [p for p in plans if p.region == 'new_tensor' or '.layers.' not in p.name]
    assert len(priority) == new_count + 1
    selected = [p.name for p in plans if '.layers.' in p.name]
    slots = max(0, 160 - len(priority))
    assert len(selected) == len(set(selected)) == slots
    if slots:
        indices = [layer_names.index(name) for name in selected]
        assert indices[0] == 0 and indices[-1] == 199
        assert set(np.diff(indices)) == {1, 2}
    assert len(plans) == max(160, new_count + 1)


@pytest.mark.parametrize('columns', [2688, 8192, 8193])
def test_extra_rows_include_values_in_last_columns(tmp_path, columns):
    for name, shape in [('a', (2, columns-128)), ('b', (3, columns))]:
        root = tmp_path / name
        root.mkdir()
        values = np.zeros(shape, dtype='float32')
        if name == 'b': values[-1, -1] = 1048576
        save_file({'matrix.weight': values}, str(root/'model.safetensors'))
    report = build_report(str(tmp_path/'a'), str(tmp_path/'b'))
    row, = [r for r in report.new_regions if r.region == 'extra_rows']
    assert row.count == (columns if columns <= 8192 else 4096)
    assert row.max_abs == 1048576
    assert not any('ほとんど 0' in f.title for f in report.flags)
    assert any('使われている' in f.title for f in report.flags)
    assert any('0 埋め' in f.title for f in report.flags)


def test_used_regions_include_sparse_nonzero_values_and_top_five(widened_pair):
    a, b = widened_pair()
    report = build_report(a, b, sample=False)
    # A rare nonzero value still means a widened region is in use, even above 99% zeros.
    regions = tuple(RegionStats('model.norm.weight', 'extra_elements', 2048, .999, float(i), True)
                    for i in range(1, 8))
    flags = make_flags(replace(report, new_regions=regions))
    used, = [f for f in flags if '使われている' in f.title]
    assert used.title.endswith('（7 領域）')
    assert [float(s.rsplit(': ', 1)[1]) for s in used.detail.split('; model.')] == [7, 6, 5, 4, 3]
    assert not any('ほとんど 0' in f.title or '0 埋め' in f.title for f in flags)
    assert 'widened regions in use: 7' in to_markdown(replace(report, new_regions=regions, sampling_enabled=True))
