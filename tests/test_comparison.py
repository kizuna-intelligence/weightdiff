import ast
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from weightdiff.application.cli import main
from weightdiff.application.collect import build_report, snapshot
from weightdiff.application.report import DISCLAIMER, to_json, to_markdown
from weightdiff.domain.compare import compare_samples, diff_config, plan_samples
from weightdiff.domain.layers import layer_type


def test_identical_models(model_factory):
    a = model_factory('a')
    b = model_factory('b')
    report = build_report(str(a), str(b))
    assert report.config_diff == []
    assert not report.tensor_diff.only_a and not report.tensor_diff.only_b
    assert not report.tensor_diff.shape_changed and not report.tensor_diff.dtype_changed
    assert not report.file_diff.only_a and not report.file_diff.only_b
    assert not report.flags
    assert report.samples and all(s.status == 'same' for s in report.samples)
    assert all(not l.added and not l.removed and not l.changed for l in report.layer_summary)
    assert report.read_stats['sample_bytes'] > 0
    assert report.read_stats['total_bytes'] == sum(report.read_stats[k] for k in ('header_bytes', 'small_file_bytes', 'sample_bytes'))
    with pytest.raises(FrozenInstanceError):
        report.a.ref = 'different'


def test_one_value_changed(model_factory):
    a = model_factory('a')
    b = model_factory('b', changed='model.embed_tokens.weight')
    report = build_report(str(a), str(b))
    assert [s.name for s in report.samples if s.status == 'different'] == ['model.embed_tokens.weight']
    assert sum(s.status == 'same' for s in report.samples) == len(report.samples) - 1
    assert len(report.flags) == 1 and '抜き取りの範囲で' in report.flags[0].title


def test_prepend_identical_layers_has_unknown_position(model_factory):
    a, b = model_factory('a'), model_factory('b', widths=(4,) * 5)
    report = build_report(str(a), str(b))
    layer, = report.layer_summary
    assert layer.b_count - layer.a_count == 2
    assert layer.added_count == 2
    assert layer.position_known is False
    assert layer.added == ()  # Do not leak SequenceMatcher's misleading tail positions.
    assert not report.tensor_diff.only_a and not report.tensor_diff.only_b
    assert report.tensor_diff.excluded_by_layer_shift == 5
    assert sum(s.layer_pair is not None for s in report.samples) == 3
    assert '位置は形からは分かりません' in to_markdown(report)


@pytest.mark.parametrize('widths,positions', [((7, 4, 4, 4), (0,)), ((4, 7, 4, 4), (1,)), ((4, 4, 4, 7), (3,))])
def test_add_distinct_layer_has_known_position(model_factory, widths, positions):
    a, b = model_factory('a'), model_factory('b', widths=widths)
    report = build_report(str(a), str(b))
    layer, = report.layer_summary
    assert layer.position_known
    assert layer.added == positions
    assert layer.added_count == 1


def test_pickle_python_and_automap_flags_in_order(model_factory):
    a = model_factory('a')
    b = model_factory('b', config={'auto_map': {'AutoModel': 'model.Custom'}})
    (b / 'model.bin').write_bytes(b'not loaded')
    (b / 'model.py').write_text('raise RuntimeError("must never execute")')
    report = build_report(str(a), str(b))
    assert report.flags[0].title == '読み込むとコードが動きうる形式'
    assert report.flags[1].title == '読み込みに trust_remote_code が要る（リポジトリのコードが動く）'
    assert [f.level for f in report.flags] == ['notice', 'notice']


@pytest.mark.parametrize('location', ['jinja', 'tokenizer', 'json'])
def test_chat_template_change(model_factory, location):
    a, b = model_factory('a'), model_factory('b')
    if location == 'tokenizer':
        (a / 'tokenizer_config.json').write_text(json.dumps({'chat_template': 'hello\nworld'}))
        (b / 'tokenizer_config.json').write_text(json.dumps({'chat_template': 'hello\nchanged'}))
    elif location == 'json':
        (a / 'chat_template.json').write_text(json.dumps({'default': 'hello\nworld'}))
        (b / 'chat_template.json').write_text(json.dumps({'default': 'hello\nchanged'}))
    else:
        (a / 'chat_template.jinja').write_text('hello\nworld')
        (b / 'chat_template.jinja').write_text('hello\nchanged')
    report = build_report(str(a), str(b))
    flag, = report.flags
    assert flag.title == '会話の書式が違う'
    assert flag.detail.startswith('1 changed lines')


def test_shards_offsets_and_full_totals(model_factory):
    a, b = model_factory('a'), model_factory('b', sharded=True)
    report = build_report(str(a), str(b))
    assert all(s.status == 'same' for s in report.samples)
    assert report.tensor_diff.a_parameters == report.tensor_diff.b_parameters == 16384 + 4 + 3*16
    assert report.tensor_diff.a_bytes == report.tensor_diff.a_parameters * 4
    assert not report.b.notes


def test_metadata_changes_and_independent_stacks(model_factory):
    a = model_factory('a', extras={'vision.layers.0.weight': np.ones((2, 2), dtype='float32')})
    b = model_factory('b', widths=(4,) * 5, extras={'vision.layers.0.weight': np.ones((3, 2), dtype='float16'), 'head.weight': np.ones(1, dtype='float32')})
    report = build_report(str(a), str(b))
    assert len(report.layer_summary) == 2
    assert report.tensor_diff.only_b == ('head.weight',)
    assert report.tensor_diff.shape_changed == ('vision.layers.0.weight',)
    assert report.tensor_diff.dtype_changed == ('vision.layers.0.weight',)
    assert report.tensor_diff.excluded_by_layer_shift == 5
    assert layer_type('x.layers.12.sub.layers.8.weight') == 'x.layers.{i}.sub.layers.8.weight'
    assert layer_type('embedding.weight') == 'embedding.weight'


def test_sampling_unavailable_is_not_reported_as_matching(model_factory):
    from weightdiff.application.report import samples_markdown
    a = model_factory('a')
    b = model_factory('b')
    report = build_report(str(a), str(b), max_tensors=0)
    assert 'limit is zero' in samples_markdown(report)
    assert 'No tensors have matching' in samples_markdown(replace(report, sample_candidates=0))


def test_progress_callback_can_be_falsey(model_factory):
    class Progress:
        def __init__(self):
            self.calls = []
        def __bool__(self):
            return False
        def __call__(self, value, desc):
            self.calls.append((value, desc))
    root = model_factory('a')
    progress = Progress()
    build_report(str(root), str(root), progress=progress)
    assert progress.calls[0][0] == 0
    assert progress.calls[-1] == (1, 'Comparison complete')
    assert any('Sampled ' in desc for _, desc in progress.calls)


def test_config_order_flattening_and_missing_values():
    a = {'z': 1, 'num_hidden_layers': 3, 'architectures': ['A'], 'auto_map': {'AutoModel': 'a.A'},
         'text_config': {'hidden_size': 5}, '_name_or_path': 'a', 'transformers_version': '1', 'removed': None}
    b = {'z': 2, 'num_hidden_layers': 5, 'architectures': ['B'], 'auto_map': {'AutoModel': 'b.B'},
         'text_config': {'hidden_size': 8}, '_name_or_path': 'b', 'transformers_version': '2', 'added': None}
    diff = diff_config(a, b)
    assert [x.key for x in diff] == ['architectures', 'num_hidden_layers', 'text_config.hidden_size', 'auto_map.AutoModel', 'added', 'removed', 'z']
    assert diff[-3].kind == 'added' and diff[-2].kind == 'removed'
    assert diff[0].a_value == '["A"]'


def test_sample_boundaries_subsampling_and_short_reads(model_factory):
    root = model_factory('a', extras={'small': np.ones(2048, dtype='float32'), 'empty': np.empty((0,), dtype='float32')})
    snap = snapshot(str(root))
    all_plans = plan_samples(snap, snap, snap.tensors)
    plans = {p.name: p for p in all_plans}
    assert plans['small'].a_ranges == ((snap.tensors['small'].data_start, 8192),)
    big = snap.tensors['model.embed_tokens.weight']
    assert plans[big.name].a_ranges == ((big.data_start, 8192), (big.data_start + big.nbytes//2 - 4096, 8192))
    names = sorted(snap.tensors)
    assert [p.name for p in plan_samples(snap, snap, names, 2)] == [names[round(k*len(names)/2)] for k in range(2)]
    assert not plan_samples(snap, snap, names, 0)
    assert compare_samples(plans['small'], [b'bad'], [b'bad']).status == 'skipped'
    assert 'unavailable' in compare_samples(plans['small'], None, None).reason


def test_no_safetensors_and_cli_outputs(model_factory, tmp_path, capsys):
    a, b = model_factory('a'), model_factory('b')
    (b / 'model.safetensors').unlink()
    (b / 'model.gguf').write_bytes(b'unread')
    report = build_report(str(a), str(b), sample=False)
    assert report.b.notes and 'safetensors が無い' in report.flags[-1].title
    assert report.samples == ()
    assert 'Sampling disabled' in to_markdown(report)
    md, js = tmp_path / 'report.md', tmp_path / 'report.json'
    assert main([str(a), str(a), '--no-sample', '--md', str(md), '--json', str(js)]) == 0
    assert md.read_text().startswith(DISCLAIMER)
    assert json.loads(js.read_text())['tensor_diff']['only_a'] == []
    assert 'Parameters' in capsys.readouterr().out


def test_dependency_boundaries():
    root = Path(__file__).resolve().parents[1] / 'src' / 'weightdiff'
    for folder, forbidden in [('domain', 'external'), ('external', 'domain')]:
        for path in (root / folder).glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert forbidden not in (node.module or '').split('.')
                elif isinstance(node, ast.Import):
                    assert all(forbidden not in alias.name.split('.') for alias in node.names)
