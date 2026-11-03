import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from .model import FingerprintCount, LayerStackSummary, ModelSnapshot

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
    result = []
    for pa, pb in sorted(pairs, key=lambda pair: (pair[0] or pair[1], pair[1] or '')):
        prefix = f'{pa} → {pb}' if pa and pb and pa != pb else pa or pb
        la, lb = stacks_a.get(pa, []), stacks_b.get(pb, [])
        names_only = (pa, pb) in name_pairs
        if a.format == 'gguf' or b.format == 'gguf':
            names_only = bool(pa and pb)
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
        known = not inserted or not all(f in ca for f in inserted)
        result.append(LayerStackSummary(
            prefix, len(la), len(lb),
            tuple(FingerprintCount(f, ca[f], cb[f]) for f in sorted(ca.keys() | cb.keys())),
            len(added), len(removed), tuple(added) if known else (), tuple(removed),
            tuple(changed), known, pa, pb, tuple(equal_pairs), 'name' if names_only else 'shape'))
    return result
