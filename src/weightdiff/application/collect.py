import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from ..domain.compare import compare_samples, diff_config, diff_files, diff_tensors, plan_samples, sample_pairs, plan_regions, region_stats, FLOAT_SIZES
from ..domain.flags import make_flags
from ..domain.layers import (summarize_layers, axis_map, insertion_runs, stack_fingerprints,
                             choose_insertion, resolve_run, layer_parts)
from ..domain.quant import (compare_quant_sample, classify_quant_samples, native_calibration_inputs,
                            plan_rearrangement, compare_rearrangement, column_group_sizes)
from ..domain.model import ModelSnapshot, Report, TensorInfo, SamplePlan
from ..external import hub, local, gguf, gguf_names, ggml
from ..external._io import start_counting, stop_counting, sample_category


def _location(ref):
    # All local/Hub and safetensors/GGUF dispatch lives in the application.
    if ref.lower().endswith('.gguf'):
        if ':' in ref and not Path(ref).is_file():
            repo, filename = ref.split(':', 1)
            if not repo or not filename:
                raise ValueError('Use repo_id:path/to/file.gguf')
            return hub, repo, filename
        path = Path(ref).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'Local GGUF file does not exist: {ref}')
        return local, str(path.parent), path.name
    return (local if Path(ref).is_dir() else hub), ref, None


def _gguf_snapshot(ref, revision, source, repo, filename):
    revision = source.resolve_revision(repo, revision)
    available = source.list_files(repo, revision)
    match = re.fullmatch(r'(.*)-(\d{5})-of-(\d{5})\.gguf', filename)
    if match:
        base, number, count = match.groups()
        if not 1 <= int(number) <= int(count):
            raise ValueError('Invalid split GGUF shard number/count')
        files = [f'{base}-{i:05d}-of-{int(count):05d}.gguf' for i in range(1, int(count) + 1)]
        if any(name not in available for name in files):
            raise ValueError('Incomplete split GGUF: not all shards are present')
    else:
        files = [filename]
    def read(name):
        size = source.file_size(repo, revision, name)
        return name, gguf.read_header(lambda length: source.read_prefix(repo, revision, name, length), size)
    tensors, metadata = {}, None
    for name, header in _parallel(read, files):
        if metadata is not None:
            arch = metadata.get('general.architecture')
            for key in ('general.architecture', f'{arch}.block_count', f'{arch}.embedding_length'):
                if metadata.get(key) != header['metadata'].get(key):
                    raise ValueError(f'Inconsistent split GGUF metadata: {key}')
        else:
            metadata = header['metadata']
        for tensor_name, item in header['tensors'].items():
            if tensor_name in tensors:
                raise ValueError(f'Duplicate GGUF tensor across shards: {tensor_name}')
            tensors[tensor_name] = TensorInfo(tensor_name, item['dtype'], item['shape'],
                                             item['nbytes'], name, item['data_start'])
    config = gguf.config_from_gguf(metadata)
    tokenizer = {'chat_template': metadata['tokenizer.chat_template']} if 'tokenizer.chat_template' in metadata else None
    return ModelSnapshot(ref, revision, tuple(files), config, tokenizer, tensors, (), 'gguf')


def _align_gguf_names(a, b):
    if a.format == b.format:
        return a, b
    hf, quantized = (a, b) if b.format == 'gguf' else (b, a)
    config = quantized.config or {}
    mapping = gguf_names.map_names(hf.tensors, quantized.tensors,
                                   config.get('model_type'), config.get('num_hidden_layers', 0))
    reverse = {gg: name for name, gg in mapping.items()}
    tensors = {}
    for original, tensor in quantized.tensors.items():
        name = reverse.get(original, original)
        if name in tensors:
            raise ValueError(f'Ambiguous GGUF name correspondence: {name}')
        tensors[name] = replace(tensor, name=name)
    quantized = replace(quantized, tensors=tensors,
        original_names={name: original for original, name in reverse.items()},
        unmapped_names=tuple(sorted(quantized.tensors.keys() - reverse.keys())))
    hf = replace(hf, unmapped_names=tuple(sorted(hf.tensors.keys() - mapping.keys())))
    return (hf, quantized) if b.format == 'gguf' else (quantized, hf)


def _parallel(fn, items):
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(copy_context().run, fn, item) for item in items]
        return [f.result() for f in futures]


def snapshot(ref, revision=None) -> ModelSnapshot:
    source, repo, filename = _location(ref)
    if filename is not None:
        return _gguf_snapshot(ref, revision, source, repo, filename)
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


def _resolve_positions(a, b, stacks, locations, notes):
    if a.format != 'safetensors' or b.format != 'safetensors': return stacks
    result = []
    for stack in stacks:
        if stack.position_known:
            result.append(stack)
            continue
        for run in insertion_runs(stack, *stack_fingerprints(a, b, stack)):
            # One matrix suffix must be present in every layer of the run.
            options = []
            for name, t in a.tensors.items():
                parts = layer_parts(name)
                if not parts or parts[0] != stack.a_prefix or parts[1] != run.a_layers[0]: continue
                if len(t.shape) != 2 or t.dtype not in FLOAT_SIZES: continue
                tensors = [a.tensors.get(f'{stack.a_prefix}{i}{parts[2]}') for i in run.a_layers]
                tensors += [b.tensors.get(f'{stack.b_prefix}{j}{parts[2]}') for j in run.b_layers]
                if all(t and len(t.shape)==2 and t.dtype in FLOAT_SIZES for t in tensors):
                    options.append((t.shape[0]*t.shape[1], parts[2]))
            if not options: continue
            _, suffix = max(options)
            acols = a.tensors[f'{stack.a_prefix}{run.a_layers[0]}{suffix}'].shape[1]
            count = min(2048, acols)
            if not count: continue
            def read(item):
                side, i = item
                snap, loc, prefix = (a, locations[0], stack.a_prefix) if side == 'a' else (b, locations[1], stack.b_prefix)
                t = snap.tensors[f'{prefix}{i}{suffix}']
                if t.shape[1] < count: raise ValueError('Insertion probe row is too narrow')
                source, repo, _ = loc
                with sample_category('position_bytes'):
                    data = source.read_ranges(repo,snap.revision,t.file,((t.data_start,count*FLOAT_SIZES[t.dtype]),))
                return (side,i), (t, data)
            try:
                probes = dict(_parallel(read,[('a',i) for i in run.a_layers]+[('b',j) for j in run.b_layers]))
                matrix = {}
                for i in run.a_layers:
                    ta, da = probes['a',i]
                    for j in run.b_layers:
                        tb, db = probes['b',j]
                        plan = SamplePlan('position probe',ta.file,((0,len(da[0])),),tb.file,((0,len(db[0])),),ta.dtype,tb.dtype)
                        matrix[i,j] = compare_samples(plan,da,db).status == 'same'
                q, matched = choose_insertion(run,matrix)
                stack = resolve_run(stack,run,q,matched)
            except Exception as exc:
                notes.append(f'Insertion position sampling unavailable: {type(exc).__name__}: {exc}')
        result.append(stack)
    return result


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
        a, b = _align_gguf_names(a, b)
        layers = summarize_layers(a, b)
        locations = _location(a.ref), _location(b.ref)
        collection_notes = []
        if sample and max_tensors:
            layers = _resolve_positions(a,b,layers,locations,collection_notes)
        widths = axis_map(a,b)
        tensors = diff_tensors(a, b, layers)
        candidates = sample_pairs(a, b, tensors, layers)
        plans = plan_samples(a, b, candidates, max_tensors, layers) if sample else []
        locations = _location(a.ref), _location(b.ref)
        def read_pair(plan):
            compare = compare_quant_sample if plan.mode == 'quant' else compare_samples
            if plan.skip_reason:
                return compare(plan, None, None)
            data = []
            for snap, location, filename, ranges in [(a, locations[0], plan.a_file, plan.a_ranges),
                                                    (b, locations[1], plan.b_file, plan.b_ranges)]:
                source, repo, _ = location
                try:
                    data.append(source.read_ranges(repo, snap.revision, filename, ranges))
                except Exception as exc:
                    data.append(f'{type(exc).__name__}: {exc}')
            if plan.mode == 'quant':
                native = {dtype: ggml.roundtrip(values, dtype)
                          for dtype, values in native_calibration_inputs(plan, *data).items()}
                return compare_quant_sample(plan, *data, roundtrips=native)
            return compare(plan, *data, width_map=widths)
        update(.3, f'Sampling {len(plans)} tensors')
        samples = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(copy_context().run, read_pair, p) for p in plans]
            for i, future in enumerate(futures):
                samples.append(future.result())
                update(.3 + .65 * (i + 1) / len(plans), f'Sampled {i+1}/{len(plans)} tensors')
        # Calibrate before additional reads; conversion/permutation checks precede
        # grouping into systematic differences.
        samples, noise = classify_quant_samples(samples, systematic=False)
        pairs = dict(candidates)
        group_sizes = column_group_sizes(a.config, b.config)
        def check_rearrangement(item):
            plan, result = item
            if result.status != 'beyond_quant_noise' or plan.mode != 'quant':
                return result
            name = plan.name.split(' → ', 1)[0]
            other = pairs.get(name, name)
            ta, tb = a.tensors[name], b.tensors[other]
            rows = plan_rearrangement(ta, tb, plan)
            if rows is None:
                return result
            try:
                reads = []
                for snap, location, filename, ranges in [
                    (a, locations[0], rows.a_file, rows.a_ranges),
                    (b, locations[1], rows.b_file, rows.b_ranges),
                ]:
                    source, repo, _ = location
                    reads.append(source.read_ranges(repo, snap.revision, filename, ranges))
                checked = compare_rearrangement(result, rows, *reads, group_sizes=group_sizes)
                if checked.status == 'rearranged' or len(ta.shape) != 2 or ta.nbytes > 32 * 1024 * 1024:
                    return checked
                source, repo, _ = locations[0]
                full_a = source.read_ranges(repo, a.revision, ta.file, ((ta.data_start, ta.nbytes),))
                return compare_rearrangement(result, rows, *reads, full_a=full_a, group_sizes=group_sizes)
            except Exception as exc:
                return replace(result, reason=result.reason +
                               f'; rearrangement check unavailable ({type(exc).__name__}: {exc})')
        update(.96, 'Checking transformations and row/column rearrangements')
        samples = _parallel(check_rearrangement, zip(plans, samples))
        samples, noise = classify_quant_samples(samples)
        regions = []
        region_plans = plan_regions(a,b,tensors,layers) if sample and max_tensors else []
        def read_region(plan):
            source, repo, _ = locations[1]
            try:
                with sample_category('region_bytes'):
                    data = source.read_ranges(repo,b.revision,plan.file,plan.ranges)
                return region_stats(plan,data), None
            except Exception as exc:
                return None, f'{plan.name} ({plan.region}): region unavailable ({type(exc).__name__}: {exc})'
        for region, note in _parallel(read_region,region_plans):
            if region is not None: regions.append(region)
            if note: collection_notes.append(note)
        if collection_notes: b = replace(b,notes=b.notes+tuple(collection_notes))
        stats = dict(counter.values)
        stats['total_bytes'] = sum(stats[k] for k in ('header_bytes', 'sample_bytes', 'small_file_bytes', 'region_bytes', 'position_bytes'))
        report = Report(a, b, diff_config(a.config, b.config), layers, tensors, tuple(samples),
                        diff_files(a.files, b.files), [], perf_counter() - started, stats, sample, len(candidates), noise, tuple(regions))
        report = replace(report, flags=make_flags(report))
        update(1, 'Comparison complete')
        return report
    finally:
        stop_counting(token)
