"""Recognize on-disk quantization metadata. Returns only dictionaries and bytes."""
import json
from math import prod

DTYPES = {'float32': 'F32', 'float16': 'F16', 'bfloat16': 'BF16'}


def logical_tensors(raw, config, read):
    result, consumed, notes = {}, set(), []
    qc = (config or {}).get('quantization_config') or {}
    method = qc.get('quant_method', '')
    if method and method not in ('bitsandbytes','gptq','awq'):
        notes.append(f'Cannot compare unsupported quantized safetensors method: {method}; packed shapes are not logical weight shapes.')
    def small(name):
        t = raw[name]
        if t['nbytes'] > 65536:
            raise ValueError('quantization metadata exceeds 64 KiB')
        return read(t['file'], ((t['data_start'], t['nbytes']),))[0]
    for name, t in raw.items():
        states = [name + '.quant_state.bitsandbytes__' + kind for kind in ('nf4', 'fp4')]
        state_name = next((n for n in states if n in raw), None)
        if state_name:
            components = {k[len(name)+1:]: v for k,v in raw.items() if k.startswith(name+'.')}
            try:
                state = json.loads(small(state_name))
                shape = state['shape']; dtype = DTYPES[state['dtype']]
                if (not isinstance(shape, list) or not shape or any(type(x) is not int or x <= 0 for x in shape)
                        or state['quant_type'] not in ('nf4','fp4') or state['blocksize'] not in (32,64,128,256,512,1024,2048,4096)):
                    raise ValueError('invalid bitsandbytes state')
                if t['nbytes'] != (prod(shape)+1)//2:
                    raise ValueError('packed byte count does not match logical shape')
                for key in ('absmax','quant_map'):
                    if key not in components: raise ValueError(f'missing {key}')
                if components['quant_map']['dtype']!='F32' or prod(components['quant_map']['shape'])!=16:
                    raise ValueError('expected 16-entry float32 quant_map')
                quant = {'method': 'bnb_'+state['quant_type'], 'state': state,
                         'components': components, 'packed': {**t, 'dtype':'U8', 'shape':[t['nbytes']]}}
                result[name] = {**t, 'shape': shape, 'dtype': dtype, 'quantization': quant,
                                'nbytes': t['nbytes']+sum(v['nbytes'] for v in components.values())}
            except Exception as exc:
                result[name] = {**t, 'quantization': {'method': 'unsupported', 'error': str(exc)}}
                notes.append(f'{name}: cannot compare quantization ({exc})')
            consumed.update(name+'.'+k for k in components)
        elif name.endswith('.weight') and name[:-6]+'SCB' in raw:
            base = name[:-6]; components = {k: raw[base+k] for k in ('SCB','weight_format') if base+k in raw}
            error = None
            if 'weight_format' in components:
                try:
                    fmt = small(base+'weight_format')
                    if int.from_bytes(fmt, 'little') != 0: error = 'unsupported bitsandbytes 8-bit weight_format (expected row-major)'
                except Exception as exc:
                    error = f'weight_format unavailable: {exc}'
            result[name] = {**t, 'dtype': DTYPES.get((config or {}).get('torch_dtype', (config or {}).get('dtype')), 'BF16'), 'quantization': {
                'method':'bnb_int8', 'packed':t, 'components':components,
                **({'error':error} if error else {})},
                'nbytes':t['nbytes']+sum(v['nbytes'] for v in components.values())}
            consumed.update(base+k for k in components)
        elif name.endswith('.qweight'):
            base=name[:-7]; components={k:raw[base+k] for k in ('qweight','qzeros','scales','g_idx') if base+k in raw}
            try:
                if method not in ('gptq','awq'): raise ValueError(f'unsupported quant_method {method!r}')
                bits=qc.get('bits',4)
                if bits not in (2,3,4,8): raise ValueError(f'unsupported packed bit width {bits}')
                if not all(k in components for k in ('qzeros','scales')): raise ValueError('missing scales or qzeros')
                if qc.get('version','gemm').lower() not in ('gemm','gptq','gptq_v2') or qc.get('checkpoint_format','gptq') not in ('gptq','gptq_v2'):
                    raise ValueError('unsupported packed checkpoint layout')
                inputs=t['shape'][0]*32//bits if method=='gptq' else t['shape'][0]
                outputs=components['scales']['shape'][1]
                if len(t['shape'])!=2 or t['dtype']!='I32':raise ValueError('expected 2D int32 qweight')
                group=qc.get('group_size',128); group=inputs if group == -1 else group
                if type(group) is not int or group<=0:raise ValueError('invalid group_size')
                pack=32//bits
                expected=[inputs*bits//32,outputs] if method=='gptq' else [inputs,outputs//pack]
                zero_words=(outputs*bits+31)//32
                ng=components['scales']['shape'][0]
                if ((outputs*bits)%32 or t['shape']!=expected or components['qzeros']['shape']!=[ng,zero_words]
                        or components['qzeros']['dtype']!='I32' or components['scales']['dtype'] not in DTYPES.values()):
                    raise ValueError('inconsistent qweight/qzeros/scales layout')
                if 'g_idx' in components and (components['g_idx']['shape']!=[inputs] or components['g_idx']['dtype']!='I32'):
                    raise ValueError('invalid g_idx metadata')
                quant={'method':method,'bits':bits,'group_size':group,'components':components,
                       'act_order':bool(qc.get('desc_act',False)), 'symmetric':qc.get('sym',False), 'zero_offset':int(method=='gptq' and qc.get('checkpoint_format','gptq')!='gptq_v2')}
                result[base+'weight']={**t,'shape':[outputs,inputs], 'dtype':(config or {}).get('torch_dtype','BF16'),
                                     'quantization':quant,'nbytes':sum(v['nbytes'] for v in components.values())}
                result[base+'weight']['dtype']=DTYPES.get(result[base+'weight']['dtype'],'BF16')
            except (ValueError,KeyError,TypeError,IndexError) as exc:
                result[base+'weight']={**t,'quantization':{'method':'unsupported','error':str(exc)}}
                notes.append(f'{name}: cannot compare quantization ({exc})')
            consumed.update(base+k for k in components)
    result.update({k:v for k,v in raw.items() if k not in consumed and k not in result})
    return result, notes
