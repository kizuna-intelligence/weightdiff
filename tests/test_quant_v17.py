"""§17: quantized head-group permutations and uniform differences."""
import json
from dataclasses import replace

import gguf
import numpy as np

from weightdiff.application.collect import build_report, snapshot
from weightdiff.application.report import metric_cards, overview_sentence, samples_markdown
from weightdiff.domain.model import SamplePlan, SampleResult
from weightdiff.domain.quant import column_group_sizes, compare_rearrangement
from weightdiff.external.gguf import config_from_gguf


def test_quantized_128_element_groups_match_despite_unstable_ranks(model_pair):
    permutation = np.array([5, 2, 7, 0, 3, 6, 1, 4])
    def permute(a, i):
        return a.reshape(len(a), 8, 128)[:, permutation].reshape(a.shape)
    report = build_report(*model_pair(permute, shape=(16, 1024)))
    assert report.samples[0].status == 'rearranged'
    assert report.samples[0].transform == 'Column rearrangement / 列の並べ替え'


def test_quantized_training_shift_is_still_a_difference(model_pair):
    report = build_report(*model_pair(lambda a, i: a * 1.02, shape=(16, 1024)))
    assert report.samples[0].status == 'beyond_quant_noise'


def test_group_sizes_use_actual_config_values_and_gguf_metadata():
    config = {'text_config': {'head_dim': 192, 'linear_key_head_dim': 96,
                              'linear_value_head_dim': 96}}
    metadata = {'general.architecture': 'qwen35', 'qwen35.ssm.state_size': 128,
                'qwen35.attention.key_length': 256, 'other.ssm.state_size': 512}
    ggconfig = config_from_gguf(metadata)
    assert ggconfig['qwen35.ssm.state_size'] == 128
    assert column_group_sizes(config, ggconfig) == (96, 128, 192, 256)
    assert column_group_sizes({'head_dim': 192}) == (192,)
    assert column_group_sizes(None, {'head_dim': True, 'linear_key_head_dim': -1}) == (64, 128, 256)


def test_custom_head_size_flows_from_snapshot_to_comparison(model_pair):
    a, b = model_pair(lambda a, i: np.roll(a, 96, axis=1), shape=(16, 384))
    with open(a + '/config.json') as file:
        config = json.load(file)
    config['text_config'] = {'linear_value_head_dim': 96}
    with open(a + '/config.json', 'w') as file:
        json.dump(config, file)
    assert build_report(a, b).samples[0].status == 'rearranged'
    assert snapshot(a).config['text_config']['linear_value_head_dim'] == 96


def test_gguf_head_size_is_preserved_in_snapshot_and_used(model_pair, tmp_path):
    a, _ = model_pair(lambda a, i: a, shape=(16, 384))
    from safetensors.numpy import load_file
    values = next(iter(load_file(a + '/model.safetensors').values()))
    target = tmp_path / 'heads.gguf'
    writer = gguf.GGUFWriter(target, 'llama')
    writer.add_block_count(1)
    writer.add_uint32('llama.ssm.state_size', 96)
    writer.add_tensor('blk.0.attn_q.weight', gguf.quants.quantize(
        np.roll(values, 96, axis=1), gguf.GGMLQuantizationType.Q8_0),
        raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    report = build_report(a, str(target))
    assert report.b.config['llama.ssm.state_size'] == 96
    assert report.samples[0].status == 'rearranged'


def test_one_group_permutation_must_fit_every_row():
    a = np.random.default_rng(3).normal(size=(3, 512)).astype(np.float32)
    b = np.array([np.roll(row, 128 * (i + 1)) for i, row in enumerate(a)])
    plan = SamplePlan('x', 'a', (), 'b', (), tensor_shape=a.shape)
    sample = SampleResult('x', 'beyond_quant_noise', '', threshold=1e-5)
    assert compare_rearrangement(sample, plan, [a.tobytes()], [b.tobytes()], group_sizes=(128,)) == sample


def test_same_a_group_cannot_be_used_twice():
    a = np.random.default_rng(3).normal(size=(3, 256)).astype(np.float32)
    b = np.tile(a[:, :128], (1, 2))
    plan = SamplePlan('x', 'a', (), 'b', (), tensor_shape=a.shape)
    sample = SampleResult('x', 'beyond_quant_noise', '', threshold=1e-5)
    assert compare_rearrangement(sample, plan, [a.tobytes()], [b.tobytes()], group_sizes=(128,)) == sample


def test_uniform_differences_count_in_card_and_overview(model_pair):
    report = build_report(*model_pair(lambda a, i: a * 1.03, count=3))
    assert {s.status for s in report.samples} == {'systematic'}
    assert all('学習し直した、または未知の変換' in s.reason for s in report.samples)
    assert 'possible conversion rewrite' not in samples_markdown(report)
    assert 'beyond 0 · uniform 3' in metric_cards(report)
    assert '>3</' in metric_cards(report)
    assert '3 tensors differ beyond quantization noise (0 isolated, 3 uniform across sampled layers)' in overview_sentence(report)
    mixed = replace(report, samples=(*report.samples, replace(report.samples[0], status='beyond_quant_noise')))
    assert 'beyond 1 · uniform 3' in metric_cards(mixed)
    assert '>4</' in metric_cards(mixed)
    assert '4 tensors differ beyond quantization noise (1 isolated, 3 uniform across sampled layers)' in overview_sentence(mixed)
