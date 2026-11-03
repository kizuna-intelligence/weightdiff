"""Decode a bounded GGUF prefix, without GGUFReader or a whole-file mmap."""
import struct
from math import prod

from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFValueType

INITIAL_HEADER = 1024 * 1024
MAX_HEADER = 256 * 1024 * 1024


class IncompleteHeader(ValueError):
    pass


class _Cursor:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, length):
        if length < 0 or self.pos + length > len(self.data):
            raise IncompleteHeader('Incomplete GGUF header')
        data = self.data[self.pos:self.pos + length]
        self.pos += length
        return data

    def number(self, fmt):
        return struct.unpack('<' + fmt, self.take(struct.calcsize('<' + fmt)))[0]

    def string(self):
        length = self.number('Q')
        if length > MAX_HEADER:
            raise ValueError('GGUF string exceeds header limit')
        return self.take(length).decode('utf-8')

    def value(self, kind, depth=0):
        if depth > 8:
            raise ValueError('GGUF metadata arrays are too deeply nested')
        kind = GGUFValueType(kind)
        formats = {GGUFValueType.UINT8: 'B', GGUFValueType.INT8: 'b',
                   GGUFValueType.UINT16: 'H', GGUFValueType.INT16: 'h',
                   GGUFValueType.UINT32: 'I', GGUFValueType.INT32: 'i',
                   GGUFValueType.FLOAT32: 'f', GGUFValueType.BOOL: '?',
                   GGUFValueType.UINT64: 'Q', GGUFValueType.INT64: 'q',
                   GGUFValueType.FLOAT64: 'd'}
        if kind in formats:
            return self.number(formats[kind])
        if kind == GGUFValueType.STRING:
            return self.string()
        if kind == GGUFValueType.ARRAY:
            subtype, size = self.number('I'), self.number('Q')
            if size > MAX_HEADER:
                raise ValueError('GGUF metadata array exceeds header limit')
            return [self.value(subtype, depth + 1) for _ in range(size)]
        raise ValueError(f'Unsupported GGUF metadata type: {kind}')


def parse_header(data, file_size=None):
    cursor = _Cursor(data)
    if cursor.take(4) != b'GGUF':
        raise ValueError('Invalid GGUF magic (only little-endian GGUF is supported)')
    version = cursor.number('I')
    if version not in (2, 3):
        raise ValueError(f'Unsupported GGUF version: {version}')
    tensor_count, metadata_count = cursor.number('Q'), cursor.number('Q')
    if tensor_count > MAX_HEADER // 24 or metadata_count > MAX_HEADER // 12:
        raise ValueError('GGUF header counts exceed limit')
    metadata = {}
    for _ in range(metadata_count):
        key = cursor.string()
        if key in metadata:
            raise ValueError(f'Duplicate GGUF metadata key: {key}')
        metadata[key] = cursor.value(cursor.number('I'))
    alignment = metadata.get('general.alignment', 32)
    if type(alignment) is not int or alignment <= 0 or alignment > MAX_HEADER or alignment & (alignment - 1):
        raise ValueError('Invalid GGUF alignment')
    tensors = {}
    for _ in range(tensor_count):
        name, ndim = cursor.string(), cursor.number('I')
        if not 1 <= ndim <= 4:
            raise ValueError(f'Invalid GGUF dimension count: {ndim}')
        shape = tuple(reversed([cursor.number('Q') for _ in range(ndim)]))
        qtype = GGMLQuantizationType(cursor.number('I'))
        offset = cursor.number('Q')
        block, size = GGML_QUANT_SIZES[qtype]
        if not all(shape) or shape[-1] % block:
            raise ValueError(f'{name}: shape is not aligned to {qtype.name} blocks')
        if name in tensors or offset % alignment:
            raise ValueError(f'{name}: duplicate tensor or unaligned offset')
        tensors[name] = dict(dtype=qtype.name, shape=shape,
                             nbytes=prod(shape) // block * size, offset=offset)
    data_start = (cursor.pos + alignment - 1) // alignment * alignment
    if data_start > MAX_HEADER:
        raise ValueError('GGUF header exceeds 256 MiB')
    if data_start > len(data):
        raise IncompleteHeader('GGUF header padding is incomplete')
    intervals = []
    for name, tensor in tensors.items():
        tensor['data_start'] = data_start + tensor.pop('offset')
        end = tensor['data_start'] + tensor['nbytes']
        if file_size is not None and end > file_size:
            raise ValueError(f'{name}: GGUF tensor extends beyond file')
        intervals.append((tensor['data_start'], end))
    if any(end > start for (_, end), (start, _) in zip(sorted(intervals), sorted(intervals)[1:])):
        raise ValueError('GGUF tensor byte ranges overlap')
    return dict(metadata=metadata, tensors=tensors, header_bytes=data_start, version=version)


def read_header(read_prefix, file_size=None):
    """The application supplies local or HTTP Range access; this parser only sees bytes."""
    length = INITIAL_HEADER
    while True:
        wanted = min(length, file_size) if file_size is not None else length
        data = read_prefix(wanted)
        try:
            return parse_header(data, file_size)
        except IncompleteHeader:
            if len(data) < wanted or (file_size is not None and wanted == file_size):
                raise ValueError('Truncated GGUF header') from None
            if length >= MAX_HEADER:
                raise ValueError('GGUF header exceeds 256 MiB') from None
            length = min(length * 2, MAX_HEADER)


def config_from_gguf(metadata):
    arch = metadata.get('general.architecture')
    config = {'model_type': arch}
    for source, target in [('block_count', 'num_hidden_layers'), ('embedding_length', 'hidden_size'),
                           ('feed_forward_length', 'intermediate_size')]:
        if f'{arch}.{source}' in metadata:
            config[target] = metadata[f'{arch}.{source}']
    vocab = metadata.get(f'{arch}.vocab_size')
    if vocab is None and isinstance(metadata.get('tokenizer.ggml.tokens'), list):
        vocab = len(metadata['tokenizer.ggml.tokens'])
    if vocab is not None:
        config['vocab_size'] = vocab
    # Preserve actual head-size metadata for application-planned permutations.
    for key, value in metadata.items():
        if key.startswith(f'{arch}.') and key.endswith(('.ssm.state_size', '.attention.key_length',
                                                       '.attention.value_length')):
            config[key] = value
    return config
