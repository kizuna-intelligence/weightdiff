"""Optional CPU libggml calibration. No model loading, GPU, or importance matrix."""
import ctypes as ct
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from threading import Lock

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

_lock = Lock()


class Unavailable(ValueError):
    pass


@lru_cache(maxsize=1)
def _library():
    try:
        package = distribution('llama-cpp-python')
    except PackageNotFoundError:
        raise Unavailable('llama-cpp-python is not installed') from None
    # Locate the shared library without importing llama_cpp (which loads inference backends).
    candidates = [p for p in package.files or () if str(p).endswith(
        ('/libggml-base.so', '/libggml-base.dylib', '/ggml-base.dll', '/libllama.so'))]
    errors = []
    for path in candidates:
        try:
            lib = ct.CDLL(str(package.locate_file(path)))
            lib.ggml_quantize_chunk.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p,
                                               ct.c_int64, ct.c_int64, ct.c_int64, ct.c_void_p]
            lib.ggml_quantize_chunk.restype = ct.c_size_t
            lib.ggml_quantize_requires_imatrix.argtypes = [ct.c_int]
            lib.ggml_quantize_requires_imatrix.restype = ct.c_bool
            lib.ggml_get_type_traits.argtypes = [ct.c_int]
            lib.ggml_get_type_traits.restype = ct.c_void_p
            lib.ggml_type_name.argtypes = [ct.c_int]
            lib.ggml_type_name.restype = ct.c_char_p
            return lib
        except (OSError, AttributeError) as exc:
            errors.append(str(exc))
    raise Unavailable('Compatible libggml is unavailable: ' + '; '.join(errors))


def _traits(lib, qtype, block, size):
    # ggml added blck_size_interleave to its public traits struct. Validate the
    # layout against gguf-py before using either function pointer.
    prefix = [('type_name', ct.c_char_p), ('blck_size', ct.c_int64)]
    suffix = [('type_size', ct.c_size_t), ('is_quantized', ct.c_bool),
              ('to_float', ct.c_void_p), ('from_float_ref', ct.c_void_p)]
    ptr = lib.ggml_get_type_traits(int(qtype))
    if not ptr:
        raise Unavailable(f'No libggml traits for {qtype.name}')
    for middle in ([], [('blck_size_interleave', ct.c_int64)]):
        class Traits(ct.Structure):
            _fields_ = prefix + middle + suffix
        traits = Traits.from_address(ptr)
        if traits.blck_size == block and traits.type_size == size and traits.to_float:
            return traits
    raise Unavailable(f'Incompatible libggml traits for {qtype.name}')


def roundtrip(values, dtype):
    """Return dict/bytes/NumPy values; callers decide whether to use a fallback."""
    try:
        qtype = GGMLQuantizationType[dtype]
        block, size = GGML_QUANT_SIZES[qtype]
        src = np.ascontiguousarray(values, dtype=np.float32).ravel()
        if not src.size or src.size % block or not np.isfinite(src).all():
            raise Unavailable(f'{dtype}: values must be finite, complete blocks')
        lib = _library()
        # Check type name too: a newer gguf enum must never reach an older C
        # implementation under a different enum value.
        name = lib.ggml_type_name(int(qtype))
        if not name or name.decode().lower() != dtype.lower():
            raise Unavailable(f'{dtype}: libggml type is unavailable')
        if lib.ggml_quantize_requires_imatrix(int(qtype)):
            raise Unavailable(f'{dtype}: libggml requires an importance matrix; none is used')
        traits = _traits(lib, qtype, block, size)
        encoded = np.empty(src.size // block * size, dtype=np.uint8)
        decoded = np.empty_like(src)
        to_float = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_void_p, ct.c_int64)(traits.to_float)
        # Serialise optional native initialization/quantization across sample workers.
        with _lock:
            written = lib.ggml_quantize_chunk(int(qtype), src.ctypes.data, encoded.ctypes.data,
                                              0, 1, src.size, None)
            if written != encoded.nbytes:
                raise Unavailable(f'{dtype}: unexpected encoded byte count')
            to_float(encoded.ctypes.data, decoded.ctypes.data, src.size)
        if not np.isfinite(decoded).all():
            raise Unavailable(f'{dtype}: non-finite libggml roundtrip')
        return {'bytes': encoded.tobytes(), 'values': decoded}
    except (Unavailable, ValueError, KeyError) as exc:
        return {'error': str(exc)}
