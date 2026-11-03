import json

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from weightdiff.application.collect import build_report, snapshot
from weightdiff.application.report import overview_sentence, to_markdown
from weightdiff.domain.compare import diff_config, plan_samples
from weightdiff.external import local
from test_v2 import float_model, rename_prefix


@pytest.mark.parametrize('changed', [False, True])
def test_wider_tensor_top_left_rows(tmp_path, changed):
    values = np.arange(5 * 3000, dtype=np.float32).reshape(5, 3000)
    wider = np.full((8, 4000), -5, dtype=np.float32)
    wider[:5, :3000] = values
    if changed:
        wider[4, 2999] = -100  # Last column of the last sampled row.
    a = float_model(tmp_path / 'a', 'F32', values)
    b = float_model(tmp_path / 'b', 'F32', wider)
    report = build_report(a, b)
    result, = report.samples
    assert result.status == ('not_top_left' if changed else 'embedded_top_left')
    sa, sb = snapshot(a), snapshot(b)
    plan, = plan_samples(sa, sb, [('weight', 'weight')])
    assert plan.a_ranges == tuple((sa.tensors['weight'].data_start + i*12000, 12000) for i in (0, 2, 4))
    assert plan.b_ranges == tuple((sb.tensors['weight'].data_start + i*16000, 12000) for i in (0, 2, 4))
    assert report.read_stats['sample_bytes'] == 72000
    assert f'top-left block of the wider B tensor: {int(not changed)} of 1.' in overview_sentence(report)
    if changed:
        assert 'ほかの置き方は調べていない' in result.reason
    assert 'other placements have not been checked' in to_markdown(report)


@pytest.mark.parametrize('count', [0, 4096, 8192, 8193, 16384])
def test_wider_vector_ranges_follow_a_length(tmp_path, count):
    values = np.ones(count, dtype=np.float32)
    wider = np.full(count + 100, -3, dtype=np.float32)
    wider[:count] = values
    a = float_model(tmp_path / 'a', 'BF16', values)
    b = float_model(tmp_path / 'b', 'F16', wider)
    report = build_report(a, b)
    assert report.samples[0].status == 'embedded_top_left'
    plan, = plan_samples(report.a, report.b, [('weight', 'weight')])
    start = report.a.tensors['weight'].data_start
    expected = (((start, count*2),) if count else ()) if count <= 8192 else (
        (start, 4096), (start + (count//2-1024)*2, 4096))
    assert plan.a_ranges == expected


def test_wider_three_dimensions_skips_without_reading(tmp_path, monkeypatch):
    a = float_model(tmp_path / 'a', 'F32', np.ones((2, 2, 2), dtype=np.float32))
    b = float_model(tmp_path / 'b', 'F32', np.ones((3, 3, 3), dtype=np.float32))
    def unexpected(*args):
        pytest.fail('Skipped 3D tensors must not fetch samples')
    monkeypatch.setattr(local, 'read_ranges', unexpected)
    report = build_report(a, b)
    assert report.samples[0].status == 'skipped'
    assert '1D and 2D' in report.samples[0].reason


def test_wider_front_layers_pair_by_names_and_sample_shifted_values(model_factory):
    a = model_factory('a')
    b = model_factory('b', widths=(8, 8, 8, 8, 8))
    path = b / 'model.safetensors'
    tensors = load_file(path)
    for i in (0, 1):
        tensors[f'model.layers.{i}.attention.weight'] = tensors.pop(f'model.layers.{i}.mlp.weight')
    save_file(tensors, str(path))
    rename_prefix(a)
    report = build_report(str(a), str(b))
    stack, = report.layer_summary
    assert stack.pairing_basis == 'name'
    assert stack.position_known and stack.added == (0, 1)
    assert stack.equal_pairs == ((0, 2), (1, 3), (2, 4))
    assert stack.added_count == 2
    assert not report.tensor_diff.only_a and not report.tensor_diff.only_b
    assert report.tensor_diff.renamed_count == 2
    layers = [s for s in report.samples if s.layer_pair]
    assert len(layers) == 3
    assert all(s.status == 'embedded_top_left' for s in layers)
    assert [(c.a_size, c.b_size, c.tensors) for c in report.tensor_diff.axis_changes] == [(4, 8, 3)]
    md = to_markdown(report)
    assert '名前で組（形が違う）' in md
    assert 'Width 4 → 8 in 3 tensors.' in md
    assert 'positions 0–1, new layer name' in md
    limited = build_report(str(a), str(b), max_tensors=2)
    assert len(limited.samples) == 2 and limited.sample_candidates == 5


@pytest.mark.parametrize('reverse', [False, True])
def test_text_config_aliases_compare_and_keep_unrelated_keys(reverse):
    a = {'text_config': {'hidden_size': 4, 'same': 1, 'only_nested': 2}, 'vision_config': {'hidden_size': 12}}
    b = {'hidden_size': 8, 'same': 1, 'only_flat': 3}
    changes = diff_config(b, a) if reverse else diff_config(a, b)
    by_key = {c.key: c for c in changes}
    width = by_key['text_config.hidden_size ↔ hidden_size']
    assert width.kind == 'changed'
    assert (width.a_value, width.b_value) == ((8, 4) if reverse else (4, 8))
    assert not any('same' in key for key in by_key)
    assert set(by_key) == {'text_config.hidden_size ↔ hidden_size', 'text_config.only_nested', 'only_flat', 'vision_config.hidden_size'}
    # Ambiguous keys on the same side must remain distinct.
    assert len(diff_config({'text_config': {'k': 1}, 'k': 2}, {'k': 3})) == 2


def test_axis_changes_count_tensors_not_duplicate_axes_and_limit_three(model_factory):
    a = model_factory('a', widths=(), extras={f'x{i}': np.ones((i, i), dtype=np.float32) for i in range(1, 5)})
    b = model_factory('b', widths=(), extras={f'x{i}': np.ones((i+10, i+10), dtype=np.float32) for i in range(1, 5)})
    report = build_report(str(a), str(b), sample=False)
    assert [(c.a_size, c.b_size, c.tensors) for c in report.tensor_diff.axis_changes] == [(1, 11, 1), (2, 12, 1), (3, 13, 1)]
