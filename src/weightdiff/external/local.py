import json
from pathlib import Path

from ._io import MAX_SMALL_FILE, SMALL_JSON, count, header_length, local_path, parse_header


def resolve_revision(repo, revision=None):
    return revision


def list_files(repo, revision=None) -> list[str]:
    root = Path(repo).resolve()
    return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.resolve().is_relative_to(root))


def _small(repo, filename):
    path = local_path(repo, filename)
    if not path.is_file():
        return None
    if path.stat().st_size > MAX_SMALL_FILE:
        raise ValueError(f'{filename}: small file exceeds 16 MiB')
    data = path.read_bytes()
    count('small_file_bytes', len(data))
    return data


def read_json(repo, revision, filename) -> dict | None:
    if filename not in SMALL_JSON:
        raise ValueError('Only named small JSON files may be read')
    data = _small(repo, filename)
    if data is None:
        return None
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError(f'{filename}: expected a JSON object')
    return result


def read_text(repo, revision, filename) -> str | None:
    if filename != 'chat_template.jinja':
        raise ValueError('Only chat_template.jinja may be read as text')
    data = _small(repo, filename)
    return None if data is None else data.decode('utf-8')


def read_header(repo, revision, filename) -> tuple[dict, int]:
    with local_path(repo, filename).open('rb') as f:
        data = f.read(8)
        count('header_bytes', len(data))
        length = header_length(data)
        data = f.read(length)
        count('header_bytes', len(data))
        if len(data) != length:
            raise ValueError('Truncated safetensors header')
    return parse_header(data), length


def read_ranges(repo, revision, filename, ranges) -> list[bytes]:
    result = []
    with local_path(repo, filename).open('rb') as f:
        for start, length in ranges:
            if start < 0 or length < 0:
                raise ValueError('Sample ranges must have nonnegative starts and lengths')
            f.seek(start)
            data = f.read(length)
            count('sample_bytes', len(data))
            if len(data) != length:
                raise ValueError('Truncated sample range')
            result.append(data)
    return result


def file_size(repo, revision, filename):
    return local_path(repo, filename).stat().st_size


def read_prefix(repo, revision, filename, length):
    with local_path(repo, filename).open('rb') as file:
        data = file.read(length)
    count('header_bytes', len(data))
    return data
