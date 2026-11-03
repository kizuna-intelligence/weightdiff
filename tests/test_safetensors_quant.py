"""Independent on-disk fixtures for logical shapes, packing and calibration."""
import json
from collections import Counter
from dataclasses import replace
import numpy as np
import pytest
from safetensors.numpy import save_file
from weightdiff.application.collect import build_report, snapshot
from weightdiff.application.report import to_markdown,metric_cards
from weightdiff.domain import safetensors_quant as codec
from weightdiff.external.safetensors_quant import logical_tensors

NF4=np.array([-1,-.6961928009986877,-.5250730514526367,-.39491748809814453,
 -.28444138169288635,-.18477343022823334,-.09105003625154495,0,.07958029955625534,
 .16093020141124725,.24611230194568634,.33791524171829224,.44070982933044434,
 .5626170039176941,.7229568362236023,1],dtype='float32')
FP4=np.array([0,.0625,8,12,4,6,2,3,-0.,-.0625,-8,-12,-4,-6,-2,-3],dtype='float32')/12


def bnb(values,name='matrix.weight',kind='nf4',block=64):
    v=values.ravel();code=NF4 if kind=='nf4' else FP4
    blocks=np.pad(v,(0,(-len(v))%block)).reshape(-1,block)
    scales=np.max(abs(blocks),axis=1)
    norm=np.divide(blocks,scales[:,None],out=np.zeros_like(blocks),where=scales[:,None]!=0)
    # Independent brute-force nearest levels; fixtures avoid midpoint ties.
    ids=np.argmin(abs(norm[...,None]-code),axis=-1).astype('uint8').ravel()[:len(v)]
    if len(ids)%2:ids=np.pad(ids,(0,1))
    state={'quant_type':kind,'blocksize':block,'dtype':'float32','shape':list(values.shape)}
    return {name:((ids[::2]<<4)|ids[1::2]).reshape(-1,1),name+'.absmax':scales,
            name+'.quant_map':code,name+'.quant_state.bitsandbytes__'+kind:
            np.frombuffer(json.dumps(state).encode(),dtype='uint8').copy()}


def write(root,tensors,config=None):
    root.mkdir();save_file(tensors,str(root/'model.safetensors'))
    (root/'config.json').write_text(json.dumps(config or {}));return str(root)


@pytest.mark.parametrize('kind',['nf4','fp4'])
def test_bnb_logical_metadata_exact_rtn_and_zero_blocks(tmp_path,kind):
    rng=np.random.default_rng(15);values=rng.normal(size=(128,128)).astype('float32');values[:2]=0
    a=write(tmp_path/'a',{'matrix.weight':values})
    b=write(tmp_path/'b',bnb(values,kind=kind))
    report=build_report(a,b)
    assert list(report.b.tensors)==['matrix.weight']
    assert report.b.tensors['matrix.weight'].shape==values.shape
    assert report.tensor_diff.b_parameters==values.size
    assert report.samples[0].status=='same_after_requant'
    assert '量子化だけの違い' in report.samples[0].reason
    assert 'Quantized values' in metric_cards(report)
    assert 'bitsandbytes' in to_markdown(report)
    assert report.read_stats['total_bytes']<values.nbytes*2


def test_high_nibble_first_and_unaligned_subset(tmp_path):
    values=np.tile(NF4,(64,8)).astype('float32')
    root=write(tmp_path/'q',bnb(values));snap=snapshot(root);t=snap.tensors['matrix.weight']
    requests,recipes=codec.read_plan(t,((1,7),))
    chunks=[]
    for fn,start,n in requests:
        with open(tmp_path/'q'/fn,'rb') as f:f.seek(start);chunks.append(f.read(n))
    assert np.array_equal(codec.decode(t,recipes,chunks),values.ravel()[1:8])


def test_nf4_training_change_is_beyond(tmp_path):
    rng=np.random.default_rng(41);a=rng.normal(size=(64,128)).astype('float32')
    report=build_report(write(tmp_path/'a',{'matrix.weight':a}),write(tmp_path/'b',bnb(a*1.5)))
    assert report.samples[0].status=='beyond_quant_noise'


def test_nf4_widened_insertion_and_sparse_new_layer(tmp_path,monkeypatch):
    rng=np.random.default_rng(19)
    a={'model.embed_tokens.weight':rng.normal(size=(192,64)).astype('float32')}
    for i in range(4):a[f'model.layers.{i}.linear.weight']=rng.normal(size=(256,64)).astype('float32')
    b={}
    for name,v in a.items():
        wide=np.zeros((v.shape[0],128),'float32');wide[:,:64]=v
        if '.layers.' in name:
            name=name.replace(f'.{int(name.split(".")[2])}.',f'.{int(name.split(".")[2])+1}.')
            b.update(bnb(wide,name))
        else:
            wide[:,-1]=1e12;b[name]=wide
    sparse=np.zeros((256,128),'float32');sparse[0,0]=1e12
    b.update(bnb(sparse,'model.layers.0.linear.weight'))
    from weightdiff.external import local
    original=local.read_ranges;before_sampling=[True];probe_bytes=[0]
    def counted(*args):
        data=original(*args)
        if before_sampling[0]:probe_bytes[0]+=sum(map(len,data))
        return data
    def progress(value,**kwargs):
        if value>=.3:before_sampling[0]=False
    monkeypatch.setattr(local,'read_ranges',counted)
    report=build_report(write(tmp_path/'a',a),write(tmp_path/'b',b),progress=progress)
    # Cache each layer probe once; constructing the match matrix performs no reads.
    assert probe_bytes[0]<4096
    stack,=report.layer_summary
    assert stack.added==(0,) and stack.position_basis=='samples' and stack.position_known
    assert len(stack.equal_pairs)==4
    assert all(s.status=='same_after_requant' and '左上' in s.reason for s in report.samples if s.layer_pair)
    assert any('ほとんど 0' in f.title for f in report.flags)
    assert any('ありえない大きさ' in f.title for f in report.flags)
    assert all('.0.' in r.name for r in report.new_regions if r.region=='new_tensor')


def affine(values,method,act_order=False,bits=4):
    outs,ins=values.shape;pack=32//bits;group=32;maximum=(1<<bits)-1
    groups=np.arange(ins)//group
    if act_order:groups=np.random.default_rng(11).permutation(groups)
    scales=np.empty((ins//group,outs),'float32');zeros=np.full_like(scales,maximum//2+1,dtype='int32')
    quant=np.empty_like(values,dtype='uint32')
    for g in range(ins//group):
        mask=groups==g;scale=np.max(abs(values[:,mask]),axis=1)/(maximum//2)
        scales[g]=scale
        quant[:,mask]=np.clip(np.rint(values[:,mask]/scale[:,None])+zeros[g,:,None],0,maximum).astype('uint32')
    if method=='gptq':
        packed=np.zeros((ins*bits//32,outs),'uint32')
        for c in range(ins):
            word,shift=divmod(c*bits,32)
            packed[word] |= (quant[:,c].astype('uint64')<<shift).astype('uint32')
            if shift+bits>32:packed[word+1] |= quant[:,c]>>(32-shift)
        stored=zeros-1
    else:
        assert bits==4
        packed=np.zeros((ins,outs//pack),'uint32');order=[0,4,1,5,2,6,3,7]
        for r in range(outs):packed[:,r//pack]|=quant[r]<<(bits*order[r%pack])
        stored=zeros
    qz=np.zeros((len(scales),outs*bits//32),'uint32')
    for r in range(outs):
        word,shift=divmod(r*bits,32) if method=='gptq' else (r//pack,bits*order[r%pack])
        qz[:,word] |= (stored[:,r].astype('uint64')<<shift).astype('uint32')
        if shift+bits>32:qz[:,word+1] |= stored[:,r].astype('uint32')>>(32-shift)
    result={'matrix.qweight':packed.view('int32'),'matrix.qzeros':qz.view('int32'),'matrix.scales':scales}
    if method=='gptq':result['matrix.g_idx']=groups.astype('int32')
    config={'torch_dtype':'float32','quantization_config':{'quant_method':method,'bits':bits,'group_size':group,'sym':True}}
    return result,config,(quant.astype('float32')-zeros[groups].T)*scales[groups].T


@pytest.mark.parametrize('method,act_order,bits',[('gptq',False,2),('gptq',False,3),('gptq',False,4),('gptq',True,4),('gptq',False,8),('awq',False,4)])
def test_gptq_awq_packing_groups_and_floor(tmp_path,method,act_order,bits):
    values=np.random.default_rng(8).normal(size=(64,128)).astype('float32')
    tensors,config,expected=affine(values,method,act_order,bits)
    root=write(tmp_path/'b',tensors,config);snap=snapshot(root);t=snap.tensors['matrix.weight']
    assert t.shape==values.shape and len(snap.tensors)==1
    g=tensors.get('matrix.g_idx')
    for row in (1,10,21):  # 3-bit zero points cross word boundaries at rows 10 and 21.
        requests,recipes=codec.read_plan(t,((128*row+3,61),),g)
        chunks=[]
        for fn,start,n in requests:
            with open(tmp_path/'b'/fn,'rb') as f:f.seek(start);chunks.append(f.read(n))
        assert np.allclose(codec.decode(t,recipes,chunks),expected[row,3:64],atol=1e-7)
    report=build_report(write(tmp_path/'a',{'matrix.weight':values}),root)
    assert report.samples[0].status=='within_quant_noise'
    assert report.quant_noise[0].source=='A'
    assert report.samples[0].threshold==pytest.approx(1.5*report.samples[0].expected_errors[0][1])


@pytest.mark.parametrize('method',['awq','gptq'])
def test_affine_retrained_values_do_not_calibrate_from_b(tmp_path,method):
    values=np.random.default_rng(71).normal(size=(64,128)).astype('float32')
    tensors,config,_=affine(values*1.4,method)
    report=build_report(write(tmp_path/'a',{'matrix.weight':values}),write(tmp_path/'b',tensors,config))
    assert report.samples[0].status=='beyond_quant_noise'


def test_bnb_int8_row_scale_and_format(tmp_path):
    values=np.random.default_rng(12).normal(size=(64,128)).astype('float32')
    scale=np.max(abs(values),axis=1)
    packed=np.rint(values/scale[:,None]*127).astype('int8')
    tensors={'matrix.weight':packed,'matrix.SCB':scale,'matrix.weight_format':np.array(0,dtype='uint8')}
    report=build_report(write(tmp_path/'a',{'matrix.weight':values}),write(tmp_path/'b',tensors,{'torch_dtype':'float32'}))
    assert report.samples[0].status=='same_after_requant'
    tensors['matrix.weight_format']=np.array(1,dtype='uint8')
    bad=write(tmp_path/'bad',tensors)
    result,=build_report(str(tmp_path/'a'),bad).samples
    assert result.status=='skipped' and 'weight_format' in result.reason


def test_truncated_auxiliary_and_unsupported_layout_are_explicit(tmp_path):
    values=np.ones((32,64),'float32');tensors=bnb(values);tensors['matrix.weight.absmax']=np.ones(1,'float32')
    report=build_report(write(tmp_path/'a',{'matrix.weight':values}),write(tmp_path/'b',tensors))
    assert report.samples[0].status=='skipped' and 'range' in report.samples[0].reason
    tensors,config,_=affine(np.ones((32,64),'float32'),'awq');config['quantization_config']['version']='gemv'
    b=write(tmp_path/'bad',tensors,config);snap=snapshot(b)
    assert any('unsupported packed checkpoint' in n for n in snap.notes)


def test_nested_absmax_decodes_but_exact_calibration_reports_unavailable(tmp_path):
    values=np.ones((32,64),'float32');tensors=bnb(values)
    statekey='matrix.weight.quant_state.bitsandbytes__nf4';state=json.loads(tensors[statekey].tobytes())
    state.update(nested_blocksize=256,nested_offset=.5)
    tensors[statekey]=np.frombuffer(json.dumps(state).encode(),dtype='uint8').copy()
    tensors['matrix.weight.absmax']=np.zeros(32,'uint8')
    tensors['matrix.weight.nested_absmax']=np.ones(1,'float32')
    tensors['matrix.weight.nested_quant_map']=np.full(256,.5,'float32')
    a=write(tmp_path/'a',{'matrix.weight':values});b=write(tmp_path/'b',tensors)
    result,=build_report(a,b).samples
    assert result.status=='skipped' and 'Nested absmax RTN' in result.reason
    t=snapshot(b).tensors['matrix.weight'];req,recipes=codec.read_plan(t,((17,81),))
    chunks=[]
    for fn,start,n in req:
        with open(tmp_path/'b'/fn,'rb') as f:f.seek(start);chunks.append(f.read(n))
    assert np.array_equal(codec.decode(t,recipes,chunks),np.ones(81))


def test_quantize_config_and_no_sample(tmp_path):
    values=np.ones((32,64),'float32');tensors,config,_=affine(values,'gptq')
    a=write(tmp_path/'a',{'matrix.weight':values});b=write(tmp_path/'b',tensors)
    (tmp_path/'b'/'quantize_config.json').write_text(json.dumps(config['quantization_config']))
    report=build_report(a,b,sample=False)
    assert not report.samples and not report.new_regions
    assert report.b.tensors['matrix.weight'].shape==values.shape


def test_zero_blocks_compare_restored_values_not_packed_codes(tmp_path):
    values=np.zeros((64,64),'float32');tensors=bnb(values)
    tensors['matrix.weight'][:]=255  # code 15 differs from RTN's zero code, still 0 after absmax=0.
    report=build_report(write(tmp_path/'a',{'matrix.weight':values}),write(tmp_path/'b',tensors))
    result,=report.samples
    assert result.status=='same_after_requant'
    assert 'zero measured RTN error' in to_markdown(report)


def test_bnb_no_sample_reads_only_state_metadata(tmp_path):
    values=np.ones((64,64),'float32')
    report=build_report(write(tmp_path/'a',{'matrix.weight':values}),write(tmp_path/'b',bnb(values)),sample=False)
    assert not report.samples and not report.new_regions
    assert report.read_stats['sample_bytes']==report.read_stats['position_bytes']==report.read_stats['region_bytes']==0


def test_auxiliaries_in_another_shard_and_index(tmp_path):
    values=np.random.default_rng(30).normal(size=(64,64)).astype('float32');tensors=bnb(values)
    a=write(tmp_path/'a',{'matrix.weight':values});b=tmp_path/'b';b.mkdir()
    weights={'matrix.weight':tensors.pop('matrix.weight')}
    save_file(weights,str(b/'weights.safetensors'));save_file(tensors,str(b/'aux.safetensors'))
    mapping={**{n:'weights.safetensors' for n in weights},**{n:'aux.safetensors' for n in tensors}}
    (b/'model.safetensors.index.json').write_text(json.dumps({'weight_map':mapping}))
    (b/'config.json').write_text('{}')
    report=build_report(a,str(b));assert report.samples[0].status=='same_after_requant'
    assert len(report.b.tensors)==1 and not report.b.notes


def test_invalid_act_order_is_skipped(tmp_path):
    values=np.ones((64,64),'float32');tensors,config,_=affine(values,'gptq')
    tensors['matrix.g_idx'][0]=99999
    result,=build_report(write(tmp_path/'a',{'matrix.weight':values}),write(tmp_path/'b',tensors,config)).samples
    assert result.status=='skipped' and 'g_idx' in result.reason


def test_nonzero_quantized_value_against_zero_reference_is_a_difference(tmp_path):
    zeros=np.zeros((64,64),'float32')
    result,=build_report(write(tmp_path/'a',{'matrix.weight':zeros}),write(tmp_path/'b',bnb(zeros+1))).samples
    assert result.status=='beyond_quant_noise' and result.relative_error is None
    assert result.max_abs_a==0 and result.max_abs_b==1


def test_act_order_calibration_uses_complete_reference_groups(tmp_path):
    values=np.random.default_rng(93).normal(size=(32,4096)).astype('float32')
    tensors,config,_=affine(values,'gptq',act_order=True)
    config['quantization_config']['desc_act']=True
    result,=build_report(write(tmp_path/'a',{'matrix.weight':values}),write(tmp_path/'b',tensors,config)).samples
    assert result.status=='within_quant_noise'
