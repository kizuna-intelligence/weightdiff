"""§16: verify a shared permutation, transformations and zero-noise thresholds."""
import gguf
import numpy as np
import pytest

from weightdiff.application.collect import build_report
from weightdiff.application.report import samples_markdown
from weightdiff.domain.model import SamplePlan, SampleResult
from weightdiff.domain.quant import classify_quant_samples, compare_rearrangement


@pytest.mark.parametrize('dtype', ['F32', 'Q8_0'])
def test_two_percent_training_shift_is_not_a_rearrangement(model_pair, dtype):
    report = build_report(*model_pair(lambda a, i: a * 1.02, dtype=dtype))
    assert report.samples[0].status == 'beyond_quant_noise'


def test_one_column_permutation_matches_all_three_complete_rows(model_pair):
    permutation = np.r_[128:256, 0:128]
    report = build_report(*model_pair(lambda a, i: a[:, permutation], dtype='F32'))
    assert report.samples[0].status == 'rearranged'
    assert report.samples[0].transform == 'Column rearrangement / 列の並べ替え'


def test_shared_column_group_permutation_with_quantization_noise():
    a = np.random.default_rng(1).normal(size=(3, 512)).astype(np.float32)
    permutation = np.r_[256:512, 0:256]
    b = gguf.quants.quantize(a[:, permutation], gguf.GGMLQuantizationType.Q8_0)
    plan = SamplePlan('x', 'a', (), 'b', (), b_dtype='Q8_0', tensor_shape=(3, 512))
    sample = SampleResult('x', 'beyond_quant_noise', '', threshold=.02)
    checked = compare_rearrangement(sample, plan, [a.tobytes()], [b.tobytes()])
    assert checked.status == 'rearranged' and 'Column' in checked.transform


def test_independent_column_permutations_per_row_are_rejected(model_pair):
    def permute(a, i):
        # Sorted row distributions are identical, but no single permutation fits.
        return np.array([np.roll(row, j + 1) for j, row in enumerate(a)])
    report = build_report(*model_pair(permute, dtype='F32'))
    assert report.samples[0].status == 'beyond_quant_noise'


def test_rows_match_without_sorting_and_use_distinct_a_rows(model_pair):
    permutation = np.random.default_rng(2).permutation(16)
    report = build_report(*model_pair(lambda a, i: a[permutation], dtype='F32'))
    assert report.samples[0].status == 'rearranged'
    assert report.samples[0].transform == 'Row rearrangement / 行の並べ替え'


def test_one_a_row_cannot_confirm_three_duplicate_b_rows(model_pair):
    report = build_report(*model_pair(lambda a, i: np.repeat(a[:1], len(a), axis=0), dtype='F32'))
    assert report.samples[0].status == 'beyond_quant_noise'


@pytest.mark.parametrize('axis', [0, 1])
def test_element_transform_combines_with_row_or_column_permutation(model_pair, axis):
    report = build_report(*model_pair(lambda a, i: -np.exp(
        np.flip(a, axis=0) if axis == 0 else np.roll(a, 128, axis=1)), dtype='F32'))
    sample = report.samples[0]
    assert sample.status == 'rearranged'
    assert sample.transform.startswith('−exp(x)＋')
    assert ('Row' if axis == 0 else 'Column') in sample.transform
    assert sample.transform in samples_markdown(report)


def test_one_dimensional_negative_exp_and_permutation(model_pair):
    report = build_report(*model_pair(lambda a, i: -np.exp(a[::-1]), dtype='F32', shape=(32,)))
    assert report.samples[0].status == 'rearranged'
    assert report.samples[0].transform == '−exp(x)＋1D rearrangement / 1 次元の並べ替え'


def test_zero_expected_error_rejects_point_one_percent_shift(model_pair):
    report = build_report(*model_pair(lambda a, i: a * 1.001, dtype='F32'))
    sample = report.samples[0]
    assert sample.status == 'beyond_quant_noise'
    assert sample.expected_errors == (('F32', 0.0),)
    assert sample.threshold == 1e-5
    assert .00099 < sample.relative_error < .00101
    assert 'threshold is 1e-5 when e_exp = 0' in samples_markdown(report)


def test_zero_floor_is_strict_after_pending_calibration_too():
    pending = SampleResult('x', 'pending_quant', '', relative_error=.001,
                           quant_types=('F32',), expected_errors=(('F32', 0.0),))
    samples, _ = classify_quant_samples([pending])
    assert samples[0].status == 'beyond_quant_noise'
    assert samples[0].threshold == 1e-5


def test_one_dimensional_sort_uses_strict_threshold_even_with_quant_noise():
    a = np.linspace(.1, 1, 32, dtype=np.float32)
    b = (a[::-1] * 1.001).astype(np.float32)
    plan = SamplePlan('x', 'a', ((0, 128),), 'b', ((0, 128),), tensor_shape=(32,))
    sample = SampleResult('x', 'beyond_quant_noise', '', threshold=.02)
    checked = compare_rearrangement(sample, plan, [a.tobytes()], [b.tobytes()])
    assert checked.status == 'beyond_quant_noise'


def test_row_assignment_can_reassign_overlapping_candidates():
    # B row 0 matches either A row; B row 1 matches only A row 0.
    a = np.array([[1, 1], [1.01, 1]], dtype=np.float32)
    b = np.array([[1.005, 1], [.995, 1]], dtype=np.float32)
    plan = SamplePlan('x', 'a', (), 'b', (), tensor_shape=(2, 2))
    sample = SampleResult('x', 'beyond_quant_noise', '', threshold=.004)
    checked = compare_rearrangement(sample, plan, [a.tobytes()], [b.tobytes()], full_a=[a.tobytes()])
    assert checked.status == 'rearranged' and 'Row' in checked.transform


def test_nonfinite_rows_cannot_confirm_rearrangement():
    a = np.ones((3, 32), dtype=np.float32)
    b = a[:, ::-1].copy()
    b[-1, 0] = np.nan
    plan = SamplePlan('x', 'a', (), 'b', (), tensor_shape=(3, 32))
    sample = SampleResult('x', 'beyond_quant_noise', '', threshold=1e-5)
    assert compare_rearrangement(sample, plan, [a.tobytes()], [b.tobytes()]) == sample
