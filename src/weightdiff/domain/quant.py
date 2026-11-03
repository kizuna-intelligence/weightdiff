"""Pure block planning, reference calibration and sampled-value classification."""
from collections import defaultdict
from dataclasses import replace
from math import lcm, prod

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize, quantize

from .layers import layer_parts, layer_type
from .model import QuantNoise, SamplePlan, SampleResult

FLOATS = {'F32', 'F16', 'BF16'}


def column_group_sizes(*configs):
    """Use only head sizes present in either model, falling back if none exist."""
    sizes = set()
    def visit(config):
        if not isinstance(config, dict):
            return
        for key, value in config.items():
            if (key.split('.')[-1] in {'head_dim', 'linear_key_head_dim', 'linear_value_head_dim'}
                    or key.endswith(('.ssm.state_size', '.attention.key_length', '.attention.value_length'))):
                if type(value) is int and value > 0:
                    sizes.add(value)
            visit(value)
    for config in configs:
        visit(config)
    return tuple(sorted(sizes)) if sizes else (64, 128, 256)


def block_layout(dtype):
    try:
        return GGML_QUANT_SIZES[GGMLQuantizationType[dtype]]
    except KeyError:
        raise ValueError(f'Unsupported GGUF dtype: {dtype}') from None


def plan_quant_sample(ta, tb, name, a_format, b_format, stack=None, layer_pair=None):
    ar, br, reason = (), (), None
    try:
        if ta.shape != tb.shape:
            raise ValueError('Quantized sampling requires equal shapes; wider quantized blocks are not checked')
        ba, sa = block_layout(ta.dtype)
        bb, sb = block_layout(tb.dtype)
        if a_format != 'gguf' and ta.dtype not in FLOATS:
            raise ValueError(f'This dtype cannot be compared: {ta.dtype}')
        if b_format != 'gguf' and tb.dtype not in FLOATS:
            raise ValueError(f'This dtype cannot be compared: {tb.dtype}')
        count, alignment = prod(ta.shape), lcm(ba, bb)
        if count % alignment or any(t.nbytes != count // block * size
                                   for t, block, size in [(ta, ba, sa), (tb, bb, sb)]):
            raise ValueError('Tensor byte size/element count does not match quantization blocks')
        elements = min(2048, count)
        elements = elements // alignment * alignment
        windows = [(0, elements)] if elements else []
        if count > elements and count >= 2048:
            middle = (count // 2 - 1024) // alignment * alignment
            if middle > 0:
                windows.append((middle, 2048))
        ar = tuple((ta.data_start + start // ba * sa, n // ba * sa) for start, n in windows)
        br = tuple((tb.data_start + start // bb * sb, n // bb * sb) for start, n in windows)
    except ValueError as exc:
        reason = str(exc)
    return SamplePlan(name, ta.file, ar, tb.file, br, ta.dtype, tb.dtype, stack, layer_pair,
                      reason, 'quant', a_format, b_format, ta.shape)


def decode_values(chunks, dtype):
    return dequantize(np.frombuffer(b''.join(chunks), dtype=np.uint8),
                      GGMLQuantizationType[dtype]).astype(np.float32).ravel()


def native_calibration_inputs(plan, bytes_a, bytes_b):
    """Calculation only; application supplies optional native roundtrips as data."""
    if plan.skip_reason or not all(isinstance(chunks, (list, tuple)) for chunks in (bytes_a, bytes_b)):
        return {}
    for chunks, ranges in [(bytes_a, plan.a_ranges), (bytes_b, plan.b_ranges)]:
        if len(chunks) != len(ranges) or any(len(data) != size for data, (_, size) in zip(chunks, ranges)):
            return {}
    types = {dtype for dtype, fmt in [(plan.a_dtype, plan.a_format), (plan.b_dtype, plan.b_format)]
             if fmt == 'gguf' and (dtype.startswith('IQ') or dtype.endswith('_K'))}
    if not types:
        return {}
    try:
        va = decode_values(bytes_a, plan.a_dtype)
        vb = decode_values(bytes_b, plan.b_dtype)
    except (NotImplementedError, ValueError, KeyError):
        return {}
    return {dtype: (vb if plan.a_format == 'gguf' and dtype == plan.a_dtype else va) for dtype in types}


def relative_error(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('inf')
    denominator = float(np.linalg.norm(a))
    numerator = float(np.linalg.norm(b - a))
    return numerator / denominator if denominator else (0.0 if numerator == 0 else float('inf'))


def _transforms(a):
    """Keep coordinates intact; invalid log entries cannot confirm a permutation."""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        yield None, a
        yield '−x', -a
        yield 'x+1', a + 1
        yield 'x−1', a - 1
        yield '−exp(x)', -np.exp(a)
        yield 'exp(x)', np.exp(a)
        yield 'log(x) (x>0)', np.where(a > 0, np.log(a), np.nan)


def _transform_errors(a, b):
    errors = []
    for name, value in _transforms(a):
        if name is None:
            continue
        if name == 'log(x) (x>0)':
            positive = a > 0
            error = relative_error(value[positive], b[positive])
        else:
            error = relative_error(value, b)
        if np.isfinite(error):
            errors.append((name, error))
    return tuple(errors)


def _noise_threshold(expected_error):
    return 1e-5 if expected_error == 0 else max(2 * expected_error, expected_error + .005)


def _classify_above_floor(sample, threshold):
    if sample.relative_error <= threshold:
        return replace(sample, status='within_quant_noise', threshold=threshold,
                       reason='Within quantization noise / 量子化の雑音の範囲'
                       + f'; e={sample.relative_error:.8g}; threshold={threshold:.8g}')
    for name, error in sample.transform_errors:
        if error <= threshold:
            return replace(sample, status='same_after_transform', threshold=threshold, transform=name,
                           reason=f'Matches after transform ({name}) / 変換（{name}）をすると同じ'
                           + f'; transformed error={error:.8g}; threshold={threshold:.8g}')
    return replace(sample, status='beyond_quant_noise', threshold=threshold,
                   reason='Differs beyond quantization noise / 量子化の雑音より大きく違う'
                   + f'; e={sample.relative_error:.8g}; threshold={threshold:.8g}')


def compare_quant_sample(plan, bytes_a, bytes_b, roundtrips=None):
    types = tuple(sorted({dtype for dtype, fmt in [(plan.a_dtype, plan.a_format),
                                                   (plan.b_dtype, plan.b_format)] if fmt == 'gguf'}))
    def result(status, reason, error=None, **kwargs):
        return SampleResult(plan.name, status, reason, plan.stack_prefix, plan.layer_pair, error, types, **kwargs)
    if plan.skip_reason:
        return result('skipped', plan.skip_reason)
    for side, chunks, ranges in [('A', bytes_a, plan.a_ranges), ('B', bytes_b, plan.b_ranges)]:
        if isinstance(chunks, str) or chunks is None:
            return result('skipped', f'{side}: {chunks or "bytes unavailable"}')
        if len(chunks) != len(ranges) or any(len(data) != size for data, (_, size) in zip(chunks, ranges)):
            return result('skipped', f'{side}: incomplete quantization blocks')
    if plan.a_format == plan.b_format == 'gguf' and plan.a_dtype == plan.b_dtype:
        return result('same' if bytes_a == bytes_b else 'different',
                      'Sampled GGUF blocks are byte-identical' if bytes_a == bytes_b else 'Sampled GGUF blocks differ')
    values = []
    for dtype, chunks in [(plan.a_dtype, bytes_a), (plan.b_dtype, bytes_b)]:
        try:
            values.append(decode_values(chunks, dtype))
        except (NotImplementedError, ValueError, KeyError) as exc:
            return result('skipped', f'Cannot dequantize {dtype}: {exc}')
    va, vb = values
    error = relative_error(va, vb)
    if not np.isfinite(error):
        return result('skipped', 'Sample counts differ, non-finite values, or original has zero norm')
    expected, packed, unavailable = [], {}, []
    for dtype in types:
        reference = vb if plan.a_format == 'gguf' and dtype == plan.a_dtype else va
        try:
            encoded = quantize(reference, GGMLQuantizationType[dtype])
            decoded = dequantize(encoded.view(np.uint8), GGMLQuantizationType[dtype])
            expected_error = relative_error(reference, decoded.ravel())
            data = encoded.tobytes()
        except (NotImplementedError, ValueError) as exc:
            native = (roundtrips or {}).get(dtype, {})
            if 'bytes' not in native:
                unavailable.append(f'{dtype}: {native.get("error", str(exc))}')
                continue
            expected_error = relative_error(reference, native['values'])
            data = native['bytes']
        if np.isfinite(expected_error):
            expected.append((dtype, expected_error))
            packed[dtype] = data
        else:
            unavailable.append(f'{dtype}: invalid reference roundtrip')
    kwargs = dict(expected_errors=tuple(expected), calibration_note='; '.join(unavailable) or None)
    if (plan.a_dtype in FLOATS and plan.a_format != 'gguf' and plan.b_format == 'gguf'
            and packed.get(plan.b_dtype) == b''.join(bytes_b)):
        return result('same_after_requant',
            'Matches the original quantized to the same dtype / 元を同じ型で量子化した物と一致', error, **kwargs)
    sample = result('pending_quant', 'Awaiting per-dtype noise calibration', error,
                    transform_errors=_transform_errors(va, vb), **kwargs)
    if len(expected) == len(types):
        expected_error = max(e for _, e in expected)
        return _classify_above_floor(sample, _noise_threshold(expected_error))
    return sample


def plan_rearrangement(ta, tb, plan):
    """1D sample windows or complete, block-aligned first/middle/last 2D rows."""
    if len(ta.shape) == 1 and ta.shape == tb.shape and all(ta.shape):
        return plan
    if len(ta.shape) != 2 or ta.shape != tb.shape or not all(ta.shape):
        return None
    rows, width = ta.shape
    ar, br = [], []
    for tensor, ranges in [(ta, ar), (tb, br)]:
        block, size = block_layout(tensor.dtype)
        if width % block:
            return None
        for row in sorted({0, rows // 2, rows - 1}):
            ranges.append((tensor.data_start + row * width // block * size, width // block * size))
    return replace(plan, a_ranges=tuple(ar), b_ranges=tuple(br))


def _distinct_row_matches(complete, b, threshold):
    """Find an injective match for B rows in A, preserving column coordinates."""
    norms = np.linalg.norm(complete.astype(np.float64), axis=1)
    matches = []
    for row in b:
        # Triangle inequality prefilter; row batches bound temporary allocations.
        row_norm = np.linalg.norm(row.astype(np.float64))
        candidates = np.flatnonzero(np.abs(norms - row_norm) <= threshold * norms)
        valid = []
        for start in range(0, len(candidates), 64):
            indices = candidates[start:start + 64]
            errors = np.linalg.norm(complete[indices].astype(np.float64) - row, axis=1)
            valid.extend(indices[errors <= threshold * norms[indices]])
        if not valid:
            return False
        matches.append(valid)
    # At most three B rows; allow reassignment when candidate sets overlap.
    assigned = {}
    def assign(i, visited):
        for j in matches[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in assigned or assign(assigned[j], visited):
                assigned[j] = i
                return True
        return False
    return all(assign(i, set()) for i in range(len(matches)))


def _rearranged(sample, kind, function):
    label = f'{function}＋{kind}' if function else kind
    return replace(sample, status='rearranged', transform=label,
                   reason='Matches after rearranging (possible conversion permutation) / '
                   '並べ替えると同じ（変換のときの並べ替えの可能性）; ' + label)


def compare_rearrangement(sample, row_plan, bytes_a, bytes_b, full_a=None, *, group_sizes=(64, 128, 256)):
    """Confirm one permutation across rows, optionally after an element transform."""
    if sample.threshold is None:
        return sample
    a = decode_values(bytes_a, row_plan.a_dtype)
    b = decode_values(bytes_b, row_plan.b_dtype)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return sample
    if len(row_plan.tensor_shape) == 1:
        for function, transformed in _transforms(a):
            if relative_error(np.sort(transformed), np.sort(b)) <= min(sample.threshold, 1e-5):
                return _rearranged(sample, '1D rearrangement / 1 次元の並べ替え', function)
        return sample
    width = row_plan.tensor_shape[-1]
    a, b = a.reshape(-1, width), b.reshape(-1, width)
    # Fit intact head-sized groups on row 0. Quantization can change individual
    # value ranks, so sorting cannot reliably infer a shared column permutation.
    for function, transformed in _transforms(a):
        if not np.isfinite(transformed).all():
            continue
        for size in group_sizes:
            if type(size) is not int or size <= 0 or width % size:
                continue
            groups = transformed[0].astype(np.float64).reshape(-1, size)
            norms = np.linalg.norm(groups, axis=1)
            selected, used = [], set()
            for target in b[0].reshape(-1, size):
                distances = np.linalg.norm(groups - target, axis=1)
                errors = np.divide(distances, norms, out=np.full_like(distances, np.inf), where=norms > 0)
                errors[(norms == 0) & (distances == 0)] = 0
                index = int(np.argmin(errors))
                if index in used:
                    break
                used.add(index)
                selected.append(index)
            if len(selected) != len(groups):
                continue
            permutation = (np.asarray(selected)[:, None] * size + np.arange(size)).ravel()
            if all(relative_error(x[permutation], y) <= sample.threshold for x, y in zip(transformed, b)):
                return _rearranged(sample, 'Column rearrangement / 列の並べ替え', function)
    if full_a is not None:
        complete = decode_values(full_a, row_plan.a_dtype).reshape(row_plan.tensor_shape)
        for function, transformed in _transforms(complete):
            if np.isfinite(transformed).all() and _distinct_row_matches(transformed, b, sample.threshold):
                return _rearranged(sample, 'Row rearrangement / 行の並べ替え', function)
    return sample


def classify_quant_samples(samples, expected_layers=None, *, systematic=True):
    """Calibrate from A where supported, with an explicit per-type B/MAD fallback."""
    observed, reference, notes = defaultdict(list), defaultdict(list), defaultdict(set)
    for sample in samples:
        if sample.relative_error is None:
            continue
        for dtype in sample.quant_types:
            observed[dtype].append(sample.relative_error)
        for dtype, error in sample.expected_errors:
            reference[dtype].append(error)
        if sample.calibration_note:
            for dtype in sample.quant_types:
                notes[dtype].add(sample.calibration_note)
    noise, fallback = [], {}
    for dtype, values in sorted(observed.items()):
        median = float(np.median(values))
        mad = float(np.median(np.abs(np.asarray(values) - median)))
        fallback[dtype] = QuantNoise(dtype, median, mad, len(values), median * 200, 'B',
                                    '; '.join(sorted(notes[dtype])))
        # Report mixed calibration modes separately so a partial native failure
        # cannot silently hide the warning for the unsupported subset.
        refs = reference[dtype]
        if refs:
            med = float(np.median(refs))
            deviation = float(np.median(np.abs(np.asarray(refs) - med)))
            noise.append(QuantNoise(dtype, med, deviation, len(refs), med * 200, 'A'))
        if len(refs) != len(values):
            noise.append(fallback[dtype])
    classified = []
    for sample in samples:
        if sample.status != 'pending_quant':
            classified.append(sample)
            continue
        thresholds = []
        expected = dict(sample.expected_errors)
        for dtype in sample.quant_types:
            if dtype in expected:
                e = expected[dtype]
                thresholds.append(_noise_threshold(e))
            else:
                floor = fallback[dtype]
                thresholds.append(max(floor.median + 6 * floor.mad, 2 * floor.median))
        classified.append(_classify_above_floor(sample, max(thresholds)))
    if not systematic:
        return tuple(classified), tuple(noise)
    groups = defaultdict(list)
    for i, sample in enumerate(classified):
        name = sample.name.split(' → ', 1)[0]
        if layer_parts(name):
            groups[layer_type(name)].append(i)
    for indices in groups.values():
        group = [classified[i] for i in indices]
        observed_layers = {layer_parts(s.name.split(' → ', 1)[0])[1] for s in group}
        if len(observed_layers) < 3 or not all(s.status == 'beyond_quant_noise' for s in group):
            continue
        values = np.array([s.relative_error for s in group])
        median = float(np.median(values))
        if median > 0 and float(np.median(np.abs(values - median))) / median < .2:
            for i in indices:
                classified[i] = replace(classified[i], status='systematic', reason=
                    'All sampled layers of this kind differ similarly (retraining or an unknown transform) / '
                    '抜き取った全部の層で同じように違う（学習し直した、または未知の変換）'
                    + f'; relative error e={classified[i].relative_error:.8g}')
    return tuple(classified), tuple(noise)
