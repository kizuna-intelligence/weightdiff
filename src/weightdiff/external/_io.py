import json
import struct
import threading
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path

MAX_HEADER = 100 * 1024 * 1024
MAX_SMALL_FILE = 16 * 1024 * 1024
SMALL_JSON = {'quantize_config.json', 'config.json', 'tokenizer_config.json', 'model.safetensors.index.json', 'chat_template.json'}
_sample_category = ContextVar('weightdiff_sample_category', default='sample_bytes')
_stats = ContextVar('weightdiff_read_stats', default=None)


class ReadCounter:
    def __init__(self):
        self.values = {'header_bytes': 0, 'sample_bytes': 0, 'small_file_bytes': 0, 'range_requests': 0, 'region_bytes': 0, 'position_bytes': 0}
        self.lock = threading.Lock()

    def add(self, key, count):
        with self.lock:
            self.values[key] += count


def count(key, amount):
    if key == 'sample_bytes': key = _sample_category.get()
    counter = _stats.get()
    if counter is not None:
        counter.add(key, amount)


def start_counting():
    counter = ReadCounter()
    return counter, _stats.set(counter)


def stop_counting(token):
    _stats.reset(token)


def header_length(data):
    if len(data) != 8:
        raise ValueError('Truncated safetensors length')
    length = struct.unpack('<Q', data)[0]
    if not 2 <= length <= MAX_HEADER:
        raise ValueError('Safetensors header length must be 2 bytes to 100 MiB')
    return length


def parse_header(data):
    header = json.loads(data)
    if not isinstance(header, dict):
        raise ValueError('Safetensors header must be a JSON object')
    return header


def local_path(repo, filename):
    root = Path(repo).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError('File is outside the model folder')
    return path


@contextmanager
def sample_category(key):
    token = _sample_category.set(key)
    try:
        yield
    finally:
        _sample_category.reset(token)
