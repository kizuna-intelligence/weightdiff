import re
from collections import Counter, defaultdict
from dataclasses import replace
from difflib import SequenceMatcher

from .model import FingerprintCount, LayerStackSummary, ModelSnapshot, Run

from .shapes import is_wider

LAYER = re.compile(r'(^|\.)(?:layers|blk)\.(\d+)\.')


def layer_parts(name):
    match = LAYER.search(name)
    if match is None:
        return None
    start, end = match.span(2)
    return name[:start], int(match.group(2)), name[end:]


def layer_type(name):
    parts = layer_parts(name)
    return name if parts is None else parts[0] + '{i}' + parts[2]


def _stacks(snapshot):
    stacks = defaultdict(lambda: defaultdict(list))
    unmapped = set(snapshot.unmapped_names)
    for name, tensor in snapshot.tensors.items():
        if name in unmapped:
            continue
        parts = layer_parts(name)
        if parts:
            prefix, number, suffix = parts
            stacks[prefix][number].append((suffix, tensor.dtype, tensor.shape))
    return {prefix: [(i, tuple(sorted(items))) for i, items in sorted(layers.items())]
            for prefix, layers in stacks.items()}


def _width_fingerprints(layers, inverse):
    # Dtype conversion is checked by value sampling, not structural alignment.
    return [(n, tuple((suffix, tuple(inverse.get(v, v) for v in shape))
                      for suffix, _, shape in fp)) for n, fp in layers]


def summarize_layers(a: ModelSnapshot, b: ModelSnapshot) -> list[LayerStackSummary]:
    stacks_a, stacks_b = _stacks(a), _stacks(b)
    common = stacks_a.keys() & stacks_b.keys()
    pairs = [(p, p) for p in sorted(common)]
    unmatched_a, unmatched_b = set(stacks_a) - common, set(stacks_b) - common
    name_pairs = set()
    def fingerprints(layers, names_only):
        return [(number, tuple(item[0] for item in fp) if names_only else fp)
                for number, fp in layers]
    for names_only in (False, True):
        candidates = []
        for pa in unmatched_a:
            ca = Counter(f for _, f in fingerprints(stacks_a[pa], names_only))
            for pb in unmatched_b:
                cb = Counter(f for _, f in fingerprints(stacks_b[pb], names_only))
                overlap = sum((ca & cb).values())
                if overlap * 2 >= min(len(stacks_a[pa]), len(stacks_b[pb])):
                    candidates.append((-overlap, pa, pb))
        for _, pa, pb in sorted(candidates):
            if pa in unmatched_a and pb in unmatched_b:
                pairs.append((pa, pb))
                if names_only:
                    name_pairs.add((pa, pb))
                unmatched_a.remove(pa)
                unmatched_b.remove(pb)
    pairs.extend((p, None) for p in sorted(unmatched_a))
    pairs.extend((None, p) for p in sorted(unmatched_b))
    widths = axis_map(a, b)
    inverse = {y: x for x, y in widths.items()}
    result = []
    for pa, pb in sorted(pairs, key=lambda pair: (pair[0] or pair[1], pair[1] or '')):
        prefix = f'{pa} → {pb}' if pa and pb and pa != pb else pa or pb
        la, lb = stacks_a.get(pa, []), stacks_b.get(pb, [])
        names_only = (pa, pb) in name_pairs
        if a.format == 'gguf' or b.format == 'gguf':
            names_only = bool(pa and pb)
        basis = 'name' if names_only else 'shape'
        if a.format == b.format == 'safetensors' and pa and pb and widths:
            raw_a, raw_b = [f for _, f in la], [f for _, f in lb]
            width_a = _width_fingerprints(la, {})
            restored = _width_fingerprints(lb, inverse)
            count = lambda x, y: sum(block.size for block in SequenceMatcher(None, x, y, autojunk=False).get_matching_blocks())
            original_count = count(raw_a, raw_b)
            if original_count < min(len(la), len(lb)) // 2 and count([f for _, f in width_a], [f for _, f in restored]) > original_count:
                la, lb, names_only, basis = width_a, restored, False, 'width'
        if names_only:
            la, lb = fingerprints(la, True), fingerprints(lb, True)
        fa, fb = [x[1] for x in la], [x[1] for x in lb]
        ca, cb = Counter(fa), Counter(fb)
        added, removed, changed, inserted, equal_pairs = [], [], [], [], []
        for tag, i, j, k, l in SequenceMatcher(None, fa, fb, autojunk=False).get_opcodes():
            if tag == 'equal':
                equal_pairs.extend((la[i + t][0], lb[k + t][0]) for t in range(j - i))
                continue
            overlap = min(j - i, l - k) if tag == 'replace' else 0
            changed.extend((la[i + t][0], lb[k + t][0]) for t in range(overlap))
            removed.extend(x[0] for x in la[i + overlap:j])
            added.extend(x[0] for x in lb[k + overlap:l])
            inserted.extend(x[1] for x in lb[k + overlap:l])
        # Repeated fingerprints can make SequenceMatcher describe one insertion
        # as an insertion plus a deletion elsewhere. Cancel equal fingerprints
        # across those opcodes before reporting the remaining added positions.
        removed_fingerprints = {number: fingerprint for number, fingerprint in la}
        retained_added, retained_fingerprints = [], []
        for number, fingerprint in zip(added, inserted):
            matching = next((n for n in removed if removed_fingerprints[n] == fingerprint), None)
            if matching is not None:
                removed.remove(matching)
            else:
                retained_added.append(number)
                retained_fingerprints.append(fingerprint)
        added, inserted = retained_added, retained_fingerprints
        known = ((not inserted or not all(f in ca for f in inserted))
                 and (not removed or not all(removed_fingerprints[n] in cb for n in removed)))
        result.append(LayerStackSummary(
            prefix, len(la), len(lb),
            tuple(FingerprintCount(f, ca[f], cb[f]) for f in sorted(ca.keys() | cb.keys())),
            len(added), len(removed), tuple(added) if known else (), tuple(removed) if known else (),
            tuple(changed), known, pa, pb, tuple(equal_pairs), basis))
    return result


def axis_map(a, b):
    candidates = defaultdict(set)
    axes_a = {v for t in a.tensors.values() for v in t.shape}
    for name, ta in a.tensors.items():
        tb = b.tensors.get(name)
        if layer_parts(name) is None and tb and is_wider(ta.shape, tb.shape):
            for x, y in zip(ta.shape, tb.shape):
                if x != y:
                    candidates[x].add(y)
    return {x: next(iter(ys)) for x, ys in candidates.items()
            if len(ys) == 1 and next(iter(ys)) not in axes_a}


def stack_fingerprints(a, b, stack):
    la = _stacks(a).get(stack.a_prefix, [])
    lb = _stacks(b).get(stack.b_prefix, [])
    if stack.pairing_basis == 'width':
        inverse = {y: x for x, y in axis_map(a, b).items()}
        la, lb = _width_fingerprints(la, {}), _width_fingerprints(lb, inverse)
    elif stack.pairing_basis == 'name':
        la, lb = [[(n, tuple(x[0] for x in fp)) for n, fp in ls] for ls in (la, lb)]
    return dict(la), dict(lb)


def insertion_runs(stack, fps_a, fps_b):
    # Recover provisional insertions even when their reported positions are hidden.
    runs = []
    for kind, small, large in [('added', fps_a, fps_b), ('removed', fps_b, fps_a)]:
        ns, nl = sorted(small), sorted(large)
        fs, fl = [small[n] for n in ns], [large[n] for n in nl]
        opcodes = SequenceMatcher(None, fs, fl, autojunk=False).get_opcodes()
        inserted = [t for tag, i, j, k, l in opcodes if tag in ('insert', 'replace')
                    for t in range(k + (min(j-i, l-k) if tag == 'replace' else 0), l)]
        seen = set()
        for p in inserted:
            s = e = p
            while s > 0 and fl[s-1] == fl[p]: s -= 1
            while e+1 < len(fl) and fl[e+1] == fl[p]: e += 1
            if s == e or (s, e) in seen: continue
            seen.add((s, e))
            if sum(s <= t <= e for t in inserted) != 1: continue
            # Map the large interval into the small sequence using the opcode anchor.
            anchor = next((i-k for tag, i, j, k, l in opcodes if k <= s < l), None)
            if anchor is None: continue
            start = s + anchor
            small_numbers = ns[start:start + e-s]
            large_numbers = nl[s:e+1]
            if start < 0 or len(small_numbers) != e-s: continue
            if any(small[n] != fl[p] for n in small_numbers): continue
            choices = []
            for offset, q in enumerate(large_numbers):
                retained = large_numbers[:offset] + large_numbers[offset+1:]
                pairs = tuple(zip(small_numbers, retained))
                if kind == 'removed': pairs = tuple((y, x) for x, y in pairs)
                choices.append((q, pairs))
            aa, bb = (small_numbers, large_numbers) if kind == 'added' else (large_numbers, small_numbers)
            runs.append(Run(kind, tuple(large_numbers), tuple(choices), tuple(aa), tuple(bb)))
    return runs


def choose_insertion(run, M):
    scored = [(sum(bool(M.get((i, j), False)) for i, j in pairs), -q, q, len(pairs))
              for q, pairs in run.candidates]
    score, _, q, total = max(scored)
    return q, score == total


def resolve_run(stack, run, q, all_matched):
    if not all_matched: return stack
    inside = set(run.a_layers), set(run.b_layers)
    equal = [(i,j) for i,j in stack.equal_pairs if i not in inside[0] and j not in inside[1]]
    equal.extend(dict(run.candidates)[q])
    added, removed = list(stack.added), list(stack.removed)
    if run.kind == 'added':
        added = [n for n in added if n not in run.positions] + [q]
    else:
        removed = [n for n in removed if n not in run.positions] + [q]
    changed = tuple((i,j) for i,j in stack.changed if i not in inside[0] and j not in inside[1])
    known = len(added) == stack.added_count and len(removed) == stack.removed_count
    return replace(stack, added=tuple(sorted(added)), removed=tuple(sorted(removed)),
                   equal_pairs=tuple(sorted(equal)), changed=changed, position_known=known,
                   position_basis='samples' if known else 'structure')
