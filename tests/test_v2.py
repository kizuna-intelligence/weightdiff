import json
from dataclasses import replace

import numpy as np
import pytest
from safetensors import TensorSpec, serialize
from safetensors.numpy import load_file, save_file

from weightdiff.application.collect import build_report, snapshot
from weightdiff.application.report import overview_sentence, samples_markdown, tensor_rows, to_markdown
from weightdiff.domain.compare import compare_samples, plan_samples
from weightdiff.domain.layers import summarize_layers
from weightdiff.external import hub, local


def rename_prefix(root, old='model.', new='model.language_model.'):
    path = root / 'model.safetensors'
    tensors = load_file(path)
    save_file({new + n[len(old):] if n.startswith(old) else n: t for n, t in tensors.items()}, str(path))


def test_prefix_pair_and_outside_names(model_factory):
    a, b = model_factory('a'), model_factory('b')
    rename_prefix(a)
    report = build_report(str(a), str(b))
    stack, = report.layer_summary
    assert (stack.a_prefix, stack.b_prefix) == ('model.language_model.layers.', 'model.layers.')
    assert stack.prefix == 'model.language_model.layers. → model.layers.'
    assert stack.equal_pairs == ((0, 0), (1, 1), (2, 2))
    assert report.tensor_diff.only_a == report.tensor_diff.only_b == ()
    assert report.tensor_diff.renamed_count == 5
    assert all(s.status == 'same' for s in report.samples)
    assert [r for r in tensor_rows(report) if r[0] == 'Name location changed'] == [
        ['Name location changed', '5 tensors', '', '', '', '']]
    assert stack.prefix in to_markdown(report)


@pytest.mark.parametrize('changed', [False, True])
def test_two_new_front_layers_and_shifted_sampling(model_factory, changed):
    a, b = model_factory('a'), model_factory('b', widths=(7, 7, 4, 4, 4))
    for root, offset in [(a, 0), (b, 2)]:
        path = root / 'model.safetensors'
        tensors = load_file(path)
        for i in range(3):
            tensors[f'model.layers.{i+offset}.mlp.weight'].fill(i+1)
        if root == b and changed:
            tensors['model.layers.3.mlp.weight'][0, 0] += 1
        save_file(tensors, str(path))
    rename_prefix(a)
    report = build_report(str(a), str(b))
    stack, = report.layer_summary
    assert (stack.a_count, stack.b_count, stack.added_count) == (3, 5, 2)
    assert stack.position_known and stack.added == (0, 1)
    assert stack.equal_pairs == ((0, 2), (1, 3), (2, 4))
    assert not report.tensor_diff.only_a and not report.tensor_diff.only_b
    layer_samples = [s for s in report.samples if s.layer_pair is not None]
    assert len(layer_samples) == 3
    assert sum(s.status == 'different' for s in layer_samples) == int(changed)
    assert {s.layer_pair for s in layer_samples} == {(0, 2), (1, 3), (2, 4)}
    assert report.tensor_diff.renamed_count == 2
    assert report.sample_candidates == 5
    assert 'positions 0–1, new layer shape' in overview_sentence(report)
    assert f'{3-int(changed)} same, {int(changed)} different' in overview_sentence(report)


def test_prefix_matching_is_greedy_one_to_one_and_thresholded(model_factory):
    def tensors(stacks):
        return {f'{prefix}.layers.{i}.weight': np.ones(width, dtype=np.float32)
                for prefix, widths in stacks.items() for i, width in enumerate(widths)}
    a = model_factory('a', widths=(), extras=tensors({'alpha': [1, 1, 1, 2], 'beta': [1, 1], 'gamma': [3, 4, 5]}))
    b = model_factory('b', widths=(), extras=tensors({'one': [1, 1, 1], 'two': [2, 2], 'three': [3, 8, 9]}))
    stacks = summarize_layers(snapshot(str(a)), snapshot(str(b)))
    paired = [(s.a_prefix, s.b_prefix) for s in stacks if s.a_prefix and s.b_prefix]
    assert paired == [('alpha.layers.', 'one.layers.'), ('beta.layers.', 'two.layers.'), ('gamma.layers.', 'three.layers.')]
    assert [s.pairing_basis for s in stacks] == ['shape', 'name', 'name']
    assert len(stacks) == 3
    # Exactly 50% is sufficient.
    only_alpha = replace(snapshot(str(a)), tensors={n: t for n, t in snapshot(str(a)).tensors.items() if n.startswith('alpha.')})
    only_two = replace(snapshot(str(b)), tensors={n: t for n, t in snapshot(str(b)).tensors.items() if n.startswith('two.')})
    stack, = summarize_layers(only_alpha, only_two)
    assert (stack.a_count, stack.b_count) == (4, 2)


def float_model(root, dtype, values):
    root.mkdir()
    if dtype == 'BF16':
        data = (values.astype('<f4').view('<u4') >> 16).astype('<u2').tobytes()
    else:
        data = values.astype('<f2' if dtype == 'F16' else '<f4').tobytes()
    buffer = np.frombuffer(data, dtype=np.uint8)
    spec = TensorSpec(dtype={'BF16': 'bfloat16', 'F16': 'float16', 'F32': 'float32'}[dtype],
                      shape=values.shape, data_ptr=buffer.ctypes.data, data_len=buffer.nbytes)
    (root / 'model.safetensors').write_bytes(bytes(serialize({'weight': spec})))
    (root / 'config.json').write_text(json.dumps({}))
    return str(root)


@pytest.mark.parametrize('a_dtype,b_dtype', [('BF16', 'F16'), ('F16', 'BF16'), ('F32', 'BF16'), ('BF16', 'F32'), ('F32', 'F16'), ('F16', 'F32')])
@pytest.mark.parametrize('change_index', [None, 0, 8192])
def test_float_conversion_and_element_ranges(tmp_path, a_dtype, b_dtype, change_index):
    values = np.full(16384, 1.003, dtype=np.float32)
    a = float_model(tmp_path / 'a', a_dtype, values)
    other = values.copy()
    if change_index is not None:
        other[change_index] = 2
    b = float_model(tmp_path / 'b', b_dtype, other)
    report = build_report(a, b)
    sample, = report.samples
    assert sample.status == ('same' if change_index is None else 'different')
    assert report.tensor_diff.dtype_changed == ('weight',)
    assert report.sample_candidates == 1
    if change_index is None:
        assert '抜き取った値は同じ（型の変換の丸めの範囲）' in sample.reason
    sa, sb = snapshot(a), snapshot(b)
    plan, = plan_samples(sa, sb, [('weight', 'weight')])
    for snap, ranges, dtype in [(sa, plan.a_ranges, a_dtype), (sb, plan.b_ranges, b_dtype)]:
        size = 4 if dtype == 'F32' else 2
        start = snap.tensors['weight'].data_start
        assert ranges == ((start, 2048*size), (start + (8192-1024)*size, 2048*size))
    assert 'within dtype-conversion rounding' in samples_markdown(report)


def test_same_dtype_uses_exact_bytes_even_inside_rounding_tolerance(tmp_path):
    a = float_model(tmp_path / 'a', 'F32', np.array([1], dtype=np.float32))
    b = float_model(tmp_path / 'b', 'F32', np.array([1.00001], dtype=np.float32))
    assert build_report(a, b).samples[0].status == 'different'


def test_unsupported_dtype_is_skipped_without_reading(model_factory, monkeypatch):
    a = model_factory('a', widths=(), extras={'integer': np.ones(10, dtype=np.int32)})
    b = model_factory('b', widths=(), extras={'integer': np.ones(10, dtype=np.int32)})
    original = local.read_ranges
    def read(repo, revision, filename, ranges):
        snap = snapshot(repo)
        assert ranges[0][0] != snap.tensors['integer'].data_start
        return original(repo, revision, filename, ranges)
    monkeypatch.setattr(local, 'read_ranges', read)
    report = build_report(str(a), str(b))
    integer = next(s for s in report.samples if s.name == 'integer')
    assert integer.status == 'skipped' and 'この型は比べられません' in integer.reason


def test_default_limit_and_zero_limit_are_visible(model_factory):
    a = model_factory('a', widths=(4,)*200)
    report = build_report(str(a), str(a))
    assert len(report.samples) == 160 and report.sample_candidates == 202
    assert '160 of 202' in samples_markdown(report)
    assert 'evenly subsampled' in samples_markdown(report)
    zero = build_report(str(a), str(a), max_tensors=0)
    assert 'limit is zero' in samples_markdown(zero)
    assert 'no values were compared' in overview_sentence(replace(zero, sampling_enabled=False)).lower()


@pytest.mark.parametrize('elements,windows', [(0, 0), (2048, 1), (8192, 1), (8193, 2)])
def test_element_window_thresholds(tmp_path, elements, windows):
    a = float_model(tmp_path / 'a', 'F32', np.ones(elements, dtype=np.float32))
    snap = snapshot(a)
    plan, = plan_samples(snap, snap, ['weight'])
    assert len(plan.a_ranges) == windows
    assert build_report(a, a).samples[0].status == 'same'


def test_hub_accepts_whole_rows_larger_than_element_window(monkeypatch):
    reads = []
    def read(repo, revision, filename, start, length, category):
        reads.append((start, length))
        return b'x'*length
    monkeypatch.setattr(hub, '_range', read)
    assert len(hub.read_ranges('owner/repo', 'rev', 'model.safetensors', [(8, 8192)])[0]) == 8192
    assert len(hub.read_ranges('owner/repo', 'rev', 'model.safetensors', [(8, 12000)])[0]) == 12000
    with pytest.raises(ValueError, match='nonnegative'):
        hub.read_ranges('owner/repo', 'rev', 'model.safetensors', [(8, -1)])
    assert reads == [(8, 8192), (8, 12000)]


@pytest.mark.parametrize('left,right,status', [(1, float('inf'), 'different'), (float('inf'), float('-inf'), 'different'), (float('nan'), 1, 'different')])
def test_nonfinite_float_samples_do_not_match_finite_or_opposite_values(tmp_path, left, right, status):
    a = float_model(tmp_path / 'a', 'BF16', np.array([left], dtype=np.float32))
    b = float_model(tmp_path / 'b', 'F32', np.array([right], dtype=np.float32))
    assert build_report(a, b).samples[0].status == status


def test_all_layer_widths_changed_pair_by_names_with_unknown_insertion(model_factory):
    a = model_factory('a', widths=(4,)*3)
    b = model_factory('b', widths=(7,)*5)
    rename_prefix(a)
    report = build_report(str(a), str(b))
    stack, = report.layer_summary
    assert stack.pairing_basis == 'name' and not stack.position_known
    assert stack.added == () and stack.added_count == 2
    assert report.sample_candidates == 5
    assert report.tensor_diff.renamed_count == 2
    assert sum(s.status == 'embedded_top_left' for s in report.samples) == 3
