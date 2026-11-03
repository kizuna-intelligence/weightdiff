from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from ..domain.compare import compare_samples, diff_config, diff_files, diff_tensors, plan_samples, sample_pairs
from ..domain.flags import make_flags
from ..domain.layers import summarize_layers
from ..domain.model import ModelSnapshot, Report, TensorInfo
from ..external import hub, local
from ..external._io import start_counting, stop_counting


def _source(ref):
    # The sole local/Hub dispatch point, shared by collection and sampling.
    return local if Path(ref).is_dir() else hub


def _parallel(fn, items):
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(copy_context().run, fn, item) for item in items]
        return [f.result() for f in futures]


def snapshot(ref, revision=None) -> ModelSnapshot:
    source = _source(ref)
    revision = source.resolve_revision(ref, revision)
    files = tuple(sorted(source.list_files(ref, revision)))
    small_names = [f for f in ('config.json', 'tokenizer_config.json', 'model.safetensors.index.json',
                              'chat_template.json', 'chat_template.jinja') if f in files]
    def read_small(name):
        reader = source.read_text if name.endswith('.jinja') else source.read_json
        return name, reader(ref, revision, name)
    small = dict(_parallel(read_small, small_names))
    config = small.get('config.json')
    tokenizer = small.get('tokenizer_config.json')
    templates = {name: small[name] for name in ('chat_template.jinja', 'chat_template.json') if name in small}
    if templates:
        tokenizer = {**(tokenizer or {}), '__weightdiff_chat_templates': templates}
    notes = []
    if config is None:
        notes.append('config.json is unavailable; configuration comparison is incomplete.')
    index = small.get('model.safetensors.index.json')
    if index is not None:
        mapping = index.get('weight_map')
        if not isinstance(mapping, dict) or not all(isinstance(v, str) and v.endswith('.safetensors') for v in mapping.values()):
            raise ValueError('Invalid safetensors index weight_map')
        shards = sorted(set(mapping.values()))
    else:
        shards = [f for f in files if f.endswith('.safetensors')]
    if not shards:
        notes.append('No safetensors files; tensor and layer comparisons are unavailable.')
    def read_head(name):
        try:
            header, length = source.read_header(ref, revision, name)
            return name, header, length, None
        except Exception as exc:
            return name, {}, 0, f'{name}: header unavailable ({type(exc).__name__}: {exc})'
    tensors = {}
    for filename, header, length, note in _parallel(read_head, shards):
        if note:
            notes.append(note)
            continue
        for name, item in header.items():
            if name == '__metadata__':
                continue
            if name in tensors:
                raise ValueError(f'Duplicate tensor name across shards: {name}')
            shape, offsets, dtype = item.get('shape'), item.get('data_offsets'), item.get('dtype')
            if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int or n < 0 for n in offsets)
                    or offsets[1] < offsets[0] or not isinstance(dtype, str)):
                raise ValueError(f'{filename}: invalid tensor metadata for {name}')
            tensors[name] = TensorInfo(name, dtype, tuple(shape), offsets[1] - offsets[0], filename, 8 + length + offsets[0])
    if index is not None:
        missing = set(index['weight_map']) - tensors.keys()
        wrong = [n for n, t in tensors.items() if index['weight_map'].get(n) != t.file]
        if missing or wrong:
            notes.append(f'Index/header mismatch: {len(missing)} missing tensors, {len(wrong)} unlisted or wrong-shard tensors.')
    return ModelSnapshot(ref, revision, files, config, tokenizer, tensors, tuple(notes))


def build_report(a_ref, b_ref, a_rev=None, b_rev=None, sample=True, max_tensors=160, progress=None) -> Report:
    if max_tensors < 0:
        raise ValueError('max_tensors must be nonnegative')
    started = perf_counter()
    counter, token = start_counting()
    def update(value, description):
        if progress is not None:
            progress(value, desc=description)
    try:
        update(0, 'Reading model A metadata and headers')
        a = snapshot(a_ref, a_rev)
        update(.15, 'Reading model B metadata and headers')
        b = snapshot(b_ref, b_rev)
        layers = summarize_layers(a, b)
        tensors = diff_tensors(a, b, layers)
        candidates = sample_pairs(a, b, tensors, layers)
        plans = plan_samples(a, b, candidates, max_tensors, layers) if sample else []
        sources = _source(a.ref), _source(b.ref)
        def read_pair(plan):
            if plan.skip_reason:
                return compare_samples(plan, None, None)
            data = []
            for snap, source, filename, ranges in [(a, sources[0], plan.a_file, plan.a_ranges),
                                                    (b, sources[1], plan.b_file, plan.b_ranges)]:
                try:
                    data.append(source.read_ranges(snap.ref, snap.revision, filename, ranges))
                except Exception as exc:
                    data.append(f'{type(exc).__name__}: {exc}')
            return compare_samples(plan, *data)
        update(.3, f'Sampling {len(plans)} tensors')
        samples = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(copy_context().run, read_pair, p) for p in plans]
            for i, future in enumerate(futures):
                samples.append(future.result())
                update(.3 + .65 * (i + 1) / len(plans), f'Sampled {i+1}/{len(plans)} tensors')
        stats = dict(counter.values)
        stats['total_bytes'] = sum(stats[k] for k in ('header_bytes', 'sample_bytes', 'small_file_bytes'))
        report = Report(a, b, diff_config(a.config, b.config), layers, tensors, tuple(samples),
                        diff_files(a.files, b.files), [], perf_counter() - started, stats, sample, len(candidates))
        report = replace(report, flags=make_flags(report))
        update(1, 'Comparison complete')
        return report
    finally:
        stop_counting(token)
