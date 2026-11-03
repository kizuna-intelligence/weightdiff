from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    file: str
    data_start: int
    quantization: dict | None = None


@dataclass(frozen=True)
class ModelSnapshot:
    ref: str
    revision: str | None
    files: tuple[str, ...]
    config: dict | None
    tokenizer_config: dict | None
    tensors: dict[str, TensorInfo]
    notes: tuple[str, ...]
    format: Literal['safetensors', 'gguf'] = 'safetensors'
    original_names: dict[str, str] = field(default_factory=dict)
    unmapped_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfigChange:
    key: str
    a_value: Any
    b_value: Any
    kind: Literal['added', 'removed', 'changed']


Fingerprint = tuple[tuple[str, str, tuple[int, ...]], ...]
WidthFingerprint = tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class FingerprintCount:
    fingerprint: Fingerprint | WidthFingerprint | tuple[str, ...]
    a_count: int
    b_count: int


@dataclass(frozen=True)
class LayerStackSummary:
    prefix: str
    a_count: int
    b_count: int
    fingerprints: tuple[FingerprintCount, ...]
    added_count: int
    removed_count: int
    added: tuple[int, ...]
    removed: tuple[int, ...]
    changed: tuple[tuple[int, int], ...]
    position_known: bool
    a_prefix: str | None = None
    b_prefix: str | None = None
    equal_pairs: tuple[tuple[int, int], ...] = ()
    pairing_basis: Literal['shape', 'name', 'width'] = 'shape'
    position_basis: Literal['structure', 'samples'] = 'structure'


@dataclass(frozen=True)
class AxisChange:
    a_size: int
    b_size: int
    tensors: int


@dataclass(frozen=True)
class TensorDiff:
    only_a: tuple[str, ...]
    only_b: tuple[str, ...]
    shape_changed: tuple[str, ...]
    dtype_changed: tuple[str, ...]
    same_meta: tuple[str, ...]
    excluded_by_layer_shift: int
    a_parameters: int
    b_parameters: int
    a_bytes: int
    b_bytes: int
    name_pairs: tuple[tuple[str, str], ...] = ()
    renamed_count: int = 0
    axis_changes: tuple[AxisChange, ...] = ()


@dataclass(frozen=True)
class SamplePlan:
    name: str
    a_file: str
    a_ranges: tuple[tuple[int, int], ...]
    b_file: str
    b_ranges: tuple[tuple[int, int], ...]
    a_dtype: str = 'F32'
    b_dtype: str = 'F32'
    stack_prefix: str | None = None
    layer_pair: tuple[int, int] | None = None
    skip_reason: str | None = None
    mode: Literal['values', 'top_left', 'quant', 'st_quant'] = 'values'
    a_format: str = 'safetensors'
    b_format: str = 'safetensors'
    tensor_shape: tuple[int, ...] = ()
    a_tensor: TensorInfo | None = None
    b_tensor: TensorInfo | None = None
    a_windows: tuple[tuple[int, int], ...] = ()
    b_windows: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class SampleResult:
    name: str
    status: Literal['same', 'different', 'skipped', 'embedded_top_left', 'not_top_left', 'same_after_requant', 'within_quant_noise', 'beyond_quant_noise', 'systematic', 'pending_quant', 'same_after_transform', 'rearranged', 'rescaled']
    reason: str
    stack_prefix: str | None = None
    layer_pair: tuple[int, int] | None = None
    relative_error: float | None = None
    quant_types: tuple[str, ...] = ()
    expected_errors: tuple[tuple[str, float], ...] = ()
    threshold: float | None = None
    transform: str | None = None
    transform_errors: tuple[tuple[str, float], ...] = ()
    calibration_note: str | None = None
    max_abs_a: float | None = None
    max_abs_b: float | None = None
    scale: float | None = None


@dataclass(frozen=True)
class QuantNoise:
    dtype: str
    median: float
    mad: float
    count: int
    undetectable_percent: float
    source: Literal['A', 'B'] = 'B'
    detail: str = ''


@dataclass(frozen=True)
class FileDiff:
    only_a: tuple[str, ...]
    only_b: tuple[str, ...]


@dataclass(frozen=True)
class Flag:
    level: Literal['info', 'notice']
    title: str
    detail: str


@dataclass(frozen=True)
class Report:
    a: ModelSnapshot
    b: ModelSnapshot
    config_diff: list[ConfigChange]
    layer_summary: list[LayerStackSummary]
    tensor_diff: TensorDiff
    samples: tuple[SampleResult, ...]
    file_diff: FileDiff
    flags: list[Flag]
    elapsed_seconds: float = 0.0
    read_stats: dict[str, int] = field(default_factory=dict)
    sampling_enabled: bool = True
    sample_candidates: int = 0
    quant_noise: tuple[QuantNoise, ...] = ()
    new_regions: tuple['RegionStats', ...] = ()


@dataclass(frozen=True)
class RegionStats:
    name: str
    region: Literal['extra_columns', 'extra_rows', 'extra_elements', 'new_tensor']
    count: int
    zero_fraction: float
    max_abs: float
    finite: bool


@dataclass(frozen=True)
class RegionPlan:
    name: str
    region: Literal['extra_columns', 'extra_rows', 'extra_elements', 'new_tensor']
    file: str
    ranges: tuple[tuple[int, int], ...]
    dtype: str
    tensor: TensorInfo | None = None
    windows: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Run:
    kind: Literal['added', 'removed']
    positions: tuple[int, ...]
    candidates: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    a_layers: tuple[int, ...]
    b_layers: tuple[int, ...]
