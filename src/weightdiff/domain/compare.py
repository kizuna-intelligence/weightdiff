import json
from collections import Counter
from math import prod

import numpy as np

from .quant import plan_quant_sample
from .layers import layer_parts
from .model import AxisChange, ConfigChange, FileDiff, SamplePlan, SampleResult, TensorDiff

PRIORITY = ('architectures', 'model_type', 'num_hidden_layers', 'text_config.num_hidden_layers',
            'hidden_size', 'text_config.hidden_size', 'intermediate_size', 'vocab_size',
            'text_config.vocab_size', 'layer_types', 'text_config.layer_types', 'auto_map', 'torch_dtype')


def _flatten(value, prefix=''):
    result = {}
    for key, item in (value or {}).items():
        if key in ('_name_or_path', 'transformers_version'):
            continue
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict) and item:
            result.update(_flatten(item, path))
        else:
            result[path] = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, list) else item
    return result


def diff_config(a, b) -> list[ConfigChange]:
    a, b = _flatten(a), _flatten(b)
    aliases = {}
    for nested, flat in ((a, b), (b, a)):
        for key in sorted(list(nested)):
            if not key.startswith('text_config.'):
                continue
            short = key[len('text_config.'):]
            if key not in flat and short in flat and short not in nested:
                label = f'{key} ↔ {short}'
                nested[label], flat[label] = nested.pop(key), flat.pop(short)
                aliases[label] = key
    def order(key):
        key = aliases.get(key, key)
        for i, important in enumerate(PRIORITY):
            if key == important or key.startswith(important + '.'):
                return i, key
        return len(PRIORITY), key
    return [ConfigChange(key, a.get(key), b.get(key),
                         'added' if key not in a else 'removed' if key not in b else 'changed')
            for key in sorted(a.keys() | b.keys(), key=order)
            if key not in a or key not in b or a[key] != b[key]]


def _name_pairs(a, b, layer_summary):
    shifted_a = {s.a_prefix for s in layer_summary if s.a_count != s.b_count}
    shifted_b = {s.b_prefix for s in layer_summary if s.a_count != s.b_count}
    def included(name, shifted):
        parts = layer_parts(name)
        return parts is None or parts[0] not in shifted
    na = {n for n in a.tensors if included(n, shifted_a)}
    nb = {n for n in b.tensors if included(n, shifted_b)}
    mappings = sorted(((s.a_prefix[:-len('layers.')], s.b_prefix[:-len('layers.')])
                       for s in layer_summary if s.a_prefix and s.b_prefix
                       and s.a_prefix != s.b_prefix), key=lambda pair: (-len(pair[0]), pair))
    pairs = [(n, n) for n in sorted(na & nb)]
    used_a, used_b = set(na & nb), set(na & nb)
    for name in sorted(na - used_a):
        for old, new in mappings:
            if name.startswith(old):
                target = new + name[len(old):]
                if target in nb and target not in used_b:
                    pairs.append((name, target))
                    used_a.add(name)
                    used_b.add(target)
                break
    excluded = (a.tensors.keys() - na) | (b.tensors.keys() - nb)
    return sorted(pairs), sorted(na - used_a), sorted(nb - used_b), len(excluded)


def diff_tensors(a, b, layer_summary) -> TensorDiff:
    pairs, only_a, only_b, excluded = _name_pairs(a, b, layer_summary)
    changes = Counter()
    for na, nb in _aligned_pairs(a, b, pairs, layer_summary):
        sa, sb = a.tensors[na].shape, b.tensors[nb].shape
        if len(sa) == len(sb):
            changes.update(set((x, y) for x, y in zip(sa, sb) if x != y))
    return TensorDiff(tuple(only_a), tuple(only_b),
        tuple(na for na, nb in pairs if a.tensors[na].shape != b.tensors[nb].shape),
        tuple(na for na, nb in pairs if a.tensors[na].dtype != b.tensors[nb].dtype),
        tuple(na for na, nb in pairs if a.tensors[na].shape == b.tensors[nb].shape
              and a.tensors[na].dtype == b.tensors[nb].dtype),
        excluded,
        sum(prod(t.shape) for t in a.tensors.values()), sum(prod(t.shape) for t in b.tensors.values()),
        sum(t.nbytes for t in a.tensors.values()), sum(t.nbytes for t in b.tensors.values()),
        tuple(pairs), sum(na != nb for na, nb in pairs),
        tuple(AxisChange(x, y, count) for (x, y), count in
              sorted(changes.items(), key=lambda item: (-item[1], item[0]))[:3]))


def _aligned_pairs(a, b, name_pairs, layer_summary):
    pairs = list(name_pairs)
    # Name comparison excludes shifted stacks; sampling uses only equal opcodes.
    for stack in layer_summary:
        if stack.a_count == stack.b_count or not stack.a_prefix or not stack.b_prefix:
            continue
        by_layer = {}
        for name in a.tensors:
            parts = layer_parts(name)
            if parts and parts[0] == stack.a_prefix:
                by_layer.setdefault(parts[1], []).append((name, parts[2]))
        for i, j in stack.equal_pairs:
            for name, suffix in by_layer.get(i, []):
                other = f'{stack.b_prefix}{j}{suffix}'
                if other in b.tensors:
                    pairs.append((name, other))
    return sorted(set(pairs))


def sample_pairs(a, b, tensor_diff, layer_summary):
    pairs = _aligned_pairs(a, b, tensor_diff.name_pairs, layer_summary)
    return [(na, nb) for na, nb in pairs if a.tensors[na].shape == b.tensors[nb].shape
            or is_wider(a.tensors[na].shape, b.tensors[nb].shape)]


def is_wider(a_shape, b_shape):
    return (len(a_shape) == len(b_shape) and a_shape != b_shape
            and all(y >= x for x, y in zip(a_shape, b_shape)))


FLOAT_SIZES = {'F32': 4, 'F16': 2, 'BF16': 2}


def plan_samples(a, b, same_meta, max_tensors=160, layer_summary=()) -> list[SamplePlan]:
    if max_tensors < 0:
        raise ValueError('max_tensors must be nonnegative')
    pairs = sorted((item, item) if isinstance(item, str) else item for item in same_meta)
    n = len(pairs)
    if not max_tensors:
        return []
    if n > max_tensors:
        pairs = [pairs[min(n - 1, round(k * n / max_tensors))] for k in range(max_tensors)]
    def ranges(t):
        size = FLOAT_SIZES[t.dtype]
        count = prod(t.shape)
        first = ((t.data_start, min(2048, count) * size),) if count else ()
        return first + (((t.data_start + (count // 2 - 1024) * size, 2048 * size),)
                        if count > 8192 else ())
    plans = []
    for na, nb in pairs:
        ta, tb = a.tensors[na], b.tensors[nb]
        top_left = is_wider(ta.shape, tb.shape)
        pa, pb = layer_parts(na), layer_parts(nb)
        stack = next((s for s in layer_summary if pa and pb and s.a_prefix == pa[0]
                      and s.b_prefix == pb[0]), None)
        if a.format == 'gguf' or b.format == 'gguf':
            plans.append(plan_quant_sample(ta, tb, na if na == nb else f'{na} → {nb}',
                a.format, b.format, stack.prefix if stack else None,
                (pa[1], pb[1]) if stack else None))
            continue
        reason = None
        if ta.dtype not in FLOAT_SIZES or tb.dtype not in FLOAT_SIZES:
            reason = 'This dtype cannot be compared / この型は比べられません'
        elif ta.shape != tb.shape and not top_left:
            reason = 'Tensor shapes differ'
        elif top_left and len(ta.shape) >= 3:
            reason = 'Top-left sampling supports only 1D and 2D tensors; other placements have not been checked'
        elif any(t.nbytes != prod(t.shape) * FLOAT_SIZES[t.dtype] for t in (ta, tb)):
            reason = 'Tensor byte size does not match shape and dtype'
        pa, pb = layer_parts(na), layer_parts(nb)
        stack = next((s for s in layer_summary if pa and pb and s.a_prefix == pa[0]
                      and s.b_prefix == pb[0]), None)
        ar, br = (), ()
        if not reason:
            if top_left and len(ta.shape) == 2:
                rows, cols = ta.shape
                selected = sorted({0, rows // 2, rows - 1}) if rows and cols else []
                ar = tuple((ta.data_start + i * cols * FLOAT_SIZES[ta.dtype],
                            cols * FLOAT_SIZES[ta.dtype]) for i in selected)
                br = tuple((tb.data_start + i * tb.shape[1] * FLOAT_SIZES[tb.dtype],
                            cols * FLOAT_SIZES[tb.dtype]) for i in selected)
            else:
                # 1D wider tensors use A's element count for both windows.
                ar = (((ta.data_start, ta.nbytes),) if ta.nbytes else ()) if (
                    top_left and prod(ta.shape) <= 8192) else ranges(ta)
                br = tuple((tb.data_start + (start - ta.data_start) // FLOAT_SIZES[ta.dtype]
                            * FLOAT_SIZES[tb.dtype], length // FLOAT_SIZES[ta.dtype]
                            * FLOAT_SIZES[tb.dtype]) for start, length in ar)
        plans.append(SamplePlan(na if na == nb else f'{na} → {nb}', ta.file,
                                ar, tb.file, br,
                                ta.dtype, tb.dtype, stack.prefix if stack else None,
                                (pa[1], pb[1]) if stack else None, reason,
                                'top_left' if top_left else 'values'))
    return plans


def _float32(data, dtype):
    if dtype == 'BF16':
        return (np.frombuffer(data, dtype='<u2').astype('<u4') << 16).view('<f4')
    return np.frombuffer(data, dtype='<f2' if dtype == 'F16' else '<f4').astype(np.float32)


def compare_samples(plan, bytes_a, bytes_b) -> SampleResult:
    def result(status, reason):
        return SampleResult(plan.name, status, reason, plan.stack_prefix, plan.layer_pair)
    if plan.skip_reason:
        return result('skipped', plan.skip_reason)
    for label, data, ranges in [('a', bytes_a, plan.a_ranges), ('b', bytes_b, plan.b_ranges)]:
        if isinstance(data, str) or data is None:
            return result('skipped', f'{label}: {data or "bytes unavailable"}')
        if len(data) != len(ranges) or any(len(chunk) != length for chunk, (_, length) in zip(data, ranges)):
            return result('skipped', f'{label}: incomplete byte ranges')
    if plan.a_dtype == plan.b_dtype:
        same = bytes_a == bytes_b
        reason = f'{sum(map(len, bytes_a)):,} sampled bytes per model are ' + ('identical' if same else 'different')
    else:
        va = _float32(b''.join(bytes_a), plan.a_dtype)
        vb = _float32(b''.join(bytes_b), plan.b_dtype)
        if va.shape != vb.shape:
            return result('skipped', 'Sample element counts differ')
        with np.errstate(invalid='ignore', over='ignore'):
            within_tolerance = np.abs(va - vb) <= 2**-7 * np.maximum(np.abs(va), np.abs(vb)) + 2**-14
            same = bool(np.all((va == vb) | (np.isfinite(va) & np.isfinite(vb) & within_tolerance)))
        reason = ('Sampled values are the same (within dtype-conversion rounding) / '
                  '抜き取った値は同じ（型の変換の丸めの範囲）' if same else 'Values differ (have been rewritten)')
    if plan.mode == 'top_left':
        return result('embedded_top_left' if same else 'not_top_left',
                      'Sampled A values found in the top-left block of B' if same else
                      'Not found in the top-left block (other placements have not been checked) / '
                      '左上の区画には見つからない（ほかの置き方は調べていない）')
    return result('same' if same else 'different', reason + '; this is not a full-weight comparison')


def diff_files(a, b) -> FileDiff:
    return FileDiff(tuple(sorted(set(a) - set(b))), tuple(sorted(set(b) - set(a))))
