import json
import re
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError, build_hf_headers
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from ._io import MAX_SMALL_FILE, SMALL_JSON, count, header_length, parse_header


def _retry(operation):
    for attempt in range(4):
        try:
            return operation()
        except Exception as error:
            response = getattr(error, 'response', None)
            status = getattr(response, 'status_code', None)
            transient = isinstance(error, (
                requests.ConnectionError, requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ReadTimeout,
                # raw.read() exposes urllib3 errors without requests' wrapping.
                ProtocolError, ReadTimeoutError,
            )) or status == 429 or (isinstance(status, int) and 500 <= status < 600)
            if not transient or attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def list_files(repo, revision=None) -> list[str]:
    return _retry(lambda: HfApi().list_repo_files(repo, revision=revision))


def resolve_revision(repo, revision=None):
    # Pin mutable branches once so headers and samples cannot cross revisions.
    return _retry(lambda: HfApi().model_info(repo, revision=revision).sha)


def _small(repo, revision, filename):
    try:
        url = hf_hub_url(repo, filename, revision=revision)
        meta = _retry(lambda: get_hf_file_metadata(url))
        if meta.size is None or meta.size > MAX_SMALL_FILE:
            raise ValueError(f'{filename}: small file exceeds 16 MiB or size is unknown')
        path = _retry(lambda: hf_hub_download(repo, filename, revision=revision))
    except EntryNotFoundError:
        return None
    if Path(path).stat().st_size > MAX_SMALL_FILE:
        raise ValueError(f'{filename}: small file exceeds 16 MiB')
    data = Path(path).read_bytes()
    count('small_file_bytes', len(data))
    return data


def read_json(repo, revision, filename) -> dict | None:
    if filename not in SMALL_JSON:
        raise ValueError('Only named small JSON files may be downloaded')
    data = _small(repo, revision, filename)
    if data is None:
        return None
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError(f'{filename}: expected a JSON object')
    return result


def read_text(repo, revision, filename) -> str | None:
    if filename != 'chat_template.jinja':
        raise ValueError('Only chat_template.jinja may be downloaded as text')
    data = _small(repo, revision, filename)
    return None if data is None else data.decode('utf-8')


def _range(repo, revision, filename, start, length, category):
    if start < 0 or length <= 0:
        raise ValueError('Invalid byte range')
    def fetch():
        # Include the range in the URL to avoid stale CDN cache entries for another range.
        url = hf_hub_url(repo, filename, revision=revision) + f'?weightdiff_range={start}-{start+length-1}'
        headers = {**build_hf_headers(), 'Range': f'bytes={start}-{start+length-1}', 'Accept-Encoding': 'identity'}
        count('range_requests', 1)
        with requests.get(url, headers=headers, timeout=(15, 60), stream=True) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise ValueError(f'Expected HTTP 206 for Range; got {response.status_code}; body not read')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', response.headers.get('Content-Range', ''))
            if not match or (int(match[1]), int(match[2])) != (start, start + length - 1):
                raise ValueError('Content-Range does not match requested bytes; body not read')
            data = response.raw.read(length + 1, decode_content=False)
            count(category, len(data))
            if len(data) != length:
                raise ValueError('Truncated or oversized Range response')
            return data
    return _retry(fetch)


def read_header(repo, revision, filename) -> tuple[dict, int]:
    length = header_length(_range(repo, revision, filename, 0, 8, 'header_bytes'))
    return parse_header(_range(repo, revision, filename, 8, length, 'header_bytes')), length


def read_ranges(repo, revision, filename, ranges) -> list[bytes]:
    if any(start < 0 or length < 0 for start, length in ranges):
        raise ValueError('Sample ranges must have nonnegative starts and lengths')
    return [_range(repo, revision, filename, start, length, 'sample_bytes') if length else b''
            for start, length in ranges]
