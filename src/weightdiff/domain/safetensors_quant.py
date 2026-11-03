"""Pure plans and codecs for bounded safetensors quantization samples."""
from dataclasses import replace
from math import prod
import numpy as np
from .model import SamplePlan, SampleResult
from .quant import relative_error

SIZES={'F32':4,'F16':2,'BF16':2,'U8':1,'I8':1,'I32':4}


def floats(data,dtype):
    if dtype=='BF16':return (np.frombuffer(data,'<u2').astype('<u4')<<16).view('<f4')
    return np.frombuffer(data,{'F32':'<f4','F16':'<f2','I8':'i1','U8':'u1','I32':'<i4'}[dtype]).astype('float32')


def cast(values,dtype):
    v=np.asarray(values,dtype='<f4')
    if dtype=='BF16':
        u=v.view('<u4');return ((u+0x7fff+((u>>16)&1))&0xffff0000).view('<f4')
    return v.astype('<f2').astype('<f4') if dtype=='F16' else v


def windows(t):
    n=prod(t.shape)
    return ((0,min(2048,n)),)+(((n//2-1024,2048),) if n>8192 else ()) if n else ()


def pair_plan(ta,tb,name,stack=None,pair=None,aw=None,bw=None):
    wider=ta.shape!=tb.shape
    if aw is None:
        if len(ta.shape)==2:
            rows,cols=ta.shape
            q=tb.quantization or ta.quantization or {}
            complete=wider or q.get('method')=='bnb_int8' or q.get('act_order')
            count=cols if complete else min(2048,cols)
            selected=sorted({0,rows//2,rows-1}) if complete else sorted({0,rows//2})
            aw=tuple((r*cols,count) for r in selected)
            bw=tuple((r*tb.shape[1],count) for r in selected)
        else:aw=windows(ta);bw=aw
    reason=None
    if ta.quantization and tb.quantization:reason='Two quantized safetensors references are not yet comparable'
    if len(ta.shape)>2 and wider:reason='Quantized top-left sampling supports 1D and 2D only'
    if any(t.dtype not in ('F32','F16','BF16') for t in (ta,tb)):reason='Unsupported logical dtype'
    return SamplePlan(name,ta.file,(),tb.file,(),ta.dtype,tb.dtype,stack,pair,reason,'st_quant',
                      tensor_shape=ta.shape,a_tensor=ta,b_tensor=tb,a_windows=aw,b_windows=bw)


def _request(t,start,count):
    size=SIZES[t['dtype']]
    if start<0 or count<0 or (start+count)*size>t['nbytes']:raise ValueError('Quantization component range exceeds tensor')
    return (t['file'],t['data_start']+start*size,count*size)


def index_request(t):
    q=t.quantization or {};g=q.get('components',{}).get('g_idx')
    if not g:return None
    if g['nbytes']>1024*1024:raise ValueError('g_idx exceeds bounded metadata limit')
    return _request(g,0,prod(g['shape']))


def read_plan(t,ws,g_idx=None):
    """Each recipe describes one logical window; never reads a whole weight tensor."""
    q=t.quantization;requests=[];recipes=[]
    def add(comp,start,count):
        requests.append(_request(comp,start,count));return len(requests)-1
    if not q:
        comp={'file':t.file,'data_start':t.data_start,'nbytes':t.nbytes,'dtype':t.dtype}
        return tuple(_request(comp,s,n) for s,n in ws),tuple({'kind':'float','data':i} for i in range(len(ws)))
    if q.get('error'):raise ValueError(q['error'])
    method=q['method'];c=q.get('components',{})
    if method.startswith('bnb_') and method!='bnb_int8':
        state=q['state'];bs=state['blocksize'];total=prod(t.shape)
        code=add(c['quant_map'],0,16)
        for s,n in ws:
            lo=s//bs*bs;hi=min(total,((s+n+bs-1)//bs)*bs)
            first,last=lo//bs,(hi+bs-1)//bs
            r={'kind':'bnb','data':add(q['packed'],lo//2,(hi-lo+1)//2),
               'absmax':add(c['absmax'],first,last-first),'code':code,'lo':lo,'hi':hi,'start':s,'count':n,'blocksize':bs}
            if 'nested_absmax' in c:
                nb=state.get('nested_blocksize',256)
                r.update(nested=add(c['nested_absmax'],first//nb,(last+nb-1)//nb-first//nb),
                         nested_code=add(c['nested_quant_map'],0,256),nested_first=first,
                         nested_base=first//nb,nested_blocksize=nb)
            recipes.append(r)
    elif method=='bnb_int8':
        cols=t.shape[1]
        for s,n in ws:
            lo=s//cols*cols;hi=min(prod(t.shape),((s+n+cols-1)//cols)*cols)
            recipes.append({'kind':'int8','data':add(q['packed'],lo,hi-lo),
                            'scales':add(c['SCB'],lo//cols,(hi-lo)//cols),
                            'lo':lo,'hi':hi,'start':s,'count':n,'blocksize':cols})
    elif method in ('gptq','awq'):
        bits=q['bits'];pack=32//bits;outs,ins=t.shape;group=q['group_size']
        if method=='awq' and bits!=4:raise ValueError('AWQ GEMM supports 4 bits only')
        for s,n in ws:
            # Split flattened windows at logical row boundaries.
            end=s+n
            while s<end:
                row,col=divmod(s,ins);count=min(end-s,ins-col)
                groups=(np.arange(col,col+count)//group if g_idx is None else g_idx[col:col+count])
                ng=c['scales']['shape'][0]
                if groups.size!=count or np.any(groups<0) or np.any(groups>=ng):raise ValueError('Invalid act_order g_idx')
                gl,gh=int(groups.min()),int(groups.max())+1
                if method=='gptq':
                    wl=col*bits//32;wh=((col+count)*bits+31)//32
                    data=add(c['qweight'],wl*outs,(wh-wl)*outs)
                else:
                    wl=col;wh=col+count;data=add(c['qweight'],wl*(outs//pack),(wh-wl)*(outs//pack))
                recipes.append({'kind':method,'data':data,'scales':add(c['scales'],gl*outs,(gh-gl)*outs),
                    'zeros':add(c['qzeros'],gl*((outs*bits+31)//32),(gh-gl)*((outs*bits+31)//32)),
                    'row':row,'col':col,'count':count,'groups':groups.tolist(),'group_lo':gl,'word_lo':wl})
                s+=count
    else:raise ValueError(f'Unsupported quantization: {method}')
    return tuple(requests),tuple(recipes)


def decode(t,recipes,chunks):
    q=t.quantization or {};c=q.get('components',{});output=[]
    def get(r,key,dtype):return floats(chunks[r[key]],dtype)
    for r in recipes:
        kind=r['kind']
        if kind=='float':output.append(floats(chunks[r['data']],t.dtype));continue
        if kind=='bnb':
            packed=np.frombuffer(chunks[r['data']],dtype='u1')
            indices=np.stack((packed>>4,packed&15),axis=1).ravel()[:r['hi']-r['lo']]
            scales=get(r,'absmax',c['absmax']['dtype'])
            if 'nested' in r:
                nc=get(r,'nested_code','F32');na=get(r,'nested','F32')
                ids=np.arange(len(scales))+r['nested_first']
                scales=nc[scales.astype(int)]*na[ids//r['nested_blocksize']-r['nested_base']]+q['state']['nested_offset']
            vals=get(r,'code','F32')[indices]*np.repeat(scales,r['blocksize'])[:len(indices)]
            vals=cast(vals,t.dtype)
            offset=r['start']-r['lo'];output.append(vals[offset:offset+r['count']])
        elif kind=='int8':
            vals=get(r,'data','I8')*np.repeat(get(r,'scales',c['SCB']['dtype']),r['blocksize'])/127
            offset=r['start']-r['lo'];output.append(cast(vals[offset:offset+r['count']],t.dtype))
        else:
            bits=q['bits'];pack=32//bits;mask=(1<<bits)-1;outs,ins=t.shape
            col=np.arange(r['col'],r['col']+r['count']);row=r['row'];gr=np.asarray(r['groups'])-r['group_lo']
            words=np.frombuffer(chunks[r['data']],'<u4')
            zeros=np.frombuffer(chunks[r['zeros']],'<u4').reshape(-1,(outs*bits+31)//32)
            scales=get(r,'scales',c['scales']['dtype']).reshape(-1,outs)
            if kind=='gptq':
                wi=col*bits//32-r['word_lo'];shift=(col*bits%32).astype('uint64')
                matrix=words.reshape(-1,outs)
                low=matrix[wi,row].astype('uint64')>>shift
                crossing=shift+bits>32
                high=matrix[np.minimum(wi+1,len(matrix)-1),row].astype('uint64')
                values=(low | np.where(crossing,high<<((32-shift)%32),0))&mask
                zw,zs=divmod(row*bits,32)
                z=zeros[gr,zw].astype('uint64')>>zs
                if zs+bits>32:z |= zeros[gr,zw+1].astype('uint64')<<(32-zs)
                z &= mask
            else:
                order=np.array([0,4,1,5,2,6,3,7]);shift=order[row%8]*4
                values=(words.reshape(-1,outs//pack)[:,row//pack]>>shift)&mask
                z=(zeros[gr,row//pack]>>shift)&mask
            output.append((values.astype('float32')-(z.astype('float32')+q['zero_offset']))*scales[gr,row])
    return np.concatenate(output) if output else np.empty(0,dtype='float32')


def roundtrip_bnb(a,t,recipes,chunks):
    q=t.quantization;c=q['components'];result=[];cursor=0
    for r in recipes:
        n=r['count'];v=cast(a[cursor:cursor+n],t.dtype);cursor+=n
        # Exact RTN needs complete reference blocks/rows, not a partial window.
        if r['start']!=r['lo'] or n!=r['hi']-r['lo']:
            raise ValueError('RTN reference window does not cover complete quantization blocks')
        bs=r['blocksize'];pad=(-n)%bs
        blocks=np.pad(v,(0,pad)).reshape(-1,bs);scales=np.max(np.abs(blocks),axis=1)
        norm=np.divide(blocks,scales[:,None],out=np.zeros_like(blocks),where=scales[:,None]!=0)
        if r['kind']=='int8':
            restored=np.rint(norm*127)*scales[:,None]/127
        else:
            if 'nested' in r:raise ValueError('Nested absmax RTN requires complete second-level reference blocks; cannot compare exactly')
            code=floats(chunks[r['code']],'F32')
            # Nearest code, preserving the on-disk map order (FP4 is not sorted).
            order=np.argsort(code,kind='stable');ordered=code[order]
            boundaries=(ordered[1:]+ordered[:-1])/2
            idx=order[np.searchsorted(boundaries,norm,side='left')]
            restored=code[idx]*scales[:,None]
        result.append(cast(restored.ravel()[:n],t.dtype))
    return np.concatenate(result) if result else np.empty(0,dtype='float32')


def rtn_floor(a,t,ws,recipes):
    """Nearest affine RTN with the same bits/group size; no calibration from B."""
    bits=t.quantization['bits'];group=t.quantization['group_size'];maximum=(1<<bits)-1
    output=[];cursor=0
    for r in recipes:
        n=r['count'];vals=a[cursor:cursor+n];cursor+=n
        restored=np.empty_like(vals);groups=np.asarray(r['groups'])
        for g in np.unique(groups):
            mask=groups==g;block=vals[mask]
            if t.quantization.get('symmetric'):
                scale=float(np.max(np.abs(block)))/((1<<(bits-1))-1)
                restored[mask]=np.clip(np.rint(block/scale),-(1<<(bits-1)),(1<<(bits-1))-1)*scale if scale else block
            else:
                low=min(0.,float(block.min()));high=max(0.,float(block.max()));scale=(high-low)/maximum
                zero=np.clip(np.rint(-low/scale),0,maximum) if scale else 0
                restored[mask]=(np.clip(np.rint(block/scale)+zero,0,maximum)-zero)*scale if scale else block
        output.append(restored)
    return relative_error(a,np.concatenate(output))


def compare(plan,a,b,requant=None,floor=None):
    q=(plan.b_tensor.quantization or plan.a_tensor.quantization)
    typ=q['method'];error=relative_error(a,b)
    maxima={'max_abs_a':float(np.max(np.abs(a))) if a.size else 0.,'max_abs_b':float(np.max(np.abs(b))) if b.size else 0.}
    def result(status,reason,**kw):
        return SampleResult(plan.name,status,reason,plan.stack_prefix,plan.layer_pair,error if np.isfinite(error) else None,(typ,),**maxima,**kw)
    where='（左上） / top-left' if plan.a_tensor.shape!=plan.b_tensor.shape else ''
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return result('skipped','Non-finite values or incompatible sample sizes')
    if requant is not None:
        if np.array_equal(requant,b):return result('same_after_requant','量子化だけの違い'+where+'; reconstructed values match exactly',expected_errors=((typ,relative_error(a,requant)),))
        threshold=max(1e-5,1.5*relative_error(a,requant))
    else:threshold=max(1e-5,1.5*floor)
    status='within_quant_noise' if requant is None and error<=threshold else 'beyond_quant_noise'
    return result(status,('量子化だけの違い（RTN の床の範囲）' if status=='within_quant_noise' else '量子化の雑音より大きく違う')+where,
                  threshold=threshold,expected_errors=((typ,floor if floor is not None else relative_error(a,requant)),))


def compare_restored(plan, a, b, recipes_b, chunks_b):
    """Calibrate from A after the application has supplied bounded restored data."""
    if plan.a_tensor.quantization:
        raise ValueError('Quantized A reference is not yet supported for RTN calibration')
    q=plan.b_tensor.quantization
    if q['method'].startswith('bnb_'):
        requant=roundtrip_bnb(a,plan.b_tensor,recipes_b,chunks_b)
        return compare(plan,a,b,requant=requant)
    floor=rtn_floor(a,plan.b_tensor,plan.a_windows,recipes_b)
    return compare(plan,a,b,floor=floor)
