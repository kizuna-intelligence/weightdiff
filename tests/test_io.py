import json
import struct
from unittest.mock import Mock

import pytest
import requests

from weightdiff.application.collect import build_report, snapshot
from weightdiff.external import hub, local
from weightdiff.external._io import MAX_HEADER
from urllib3.exceptions import ProtocolError, ReadTimeoutError


def response(status=206, data=b'abcdefgh', content_range='bytes 0-7/100'):
    result = Mock()
    result.status_code = status
    result.headers = {'Content-Range': content_range}
    result.raw.read.return_value = data
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


def test_range_ignored_never_reads_body(monkeypatch):
    result = response(status=200)
    monkeypatch.setattr(hub.requests, 'get', Mock(return_value=result))
    with pytest.raises(ValueError, match='Expected HTTP 206'):
        hub.read_ranges('owner/repo', 'rev', 'model.safetensors', [(0, 8)])
    result.raw.read.assert_not_called()


def test_wrong_or_short_content_range(monkeypatch):
    result = response(content_range='bytes 10-17/100')
    monkeypatch.setattr(hub.requests, 'get', Mock(return_value=result))
    with pytest.raises(ValueError, match='Content-Range'):
        hub.read_ranges('owner/repo', 'rev', 'model.safetensors', [(0, 8)])
    result.raw.read.assert_not_called()
    result = response(data=b'abc')
    monkeypatch.setattr(hub.requests, 'get', Mock(return_value=result))
    with pytest.raises(ValueError, match='Truncated'):
        hub.read_ranges('owner/repo', 'rev', 'model.safetensors', [(0, 8)])


def test_separate_range_requests_with_auth_headers(monkeypatch):
    results = [response(), response(data=b'12345678', content_range='bytes 16-23/100')]
    get = Mock(side_effect=results)
    monkeypatch.setattr(hub.requests, 'get', get)
    monkeypatch.setattr(hub, 'build_hf_headers', lambda: {'Authorization': 'Bearer test-token'})
    assert hub.read_ranges('owner/repo', 'rev', 'model.safetensors', [(0, 8), (16, 8)]) == [b'abcdefgh', b'12345678']
    assert get.call_count == 2
    assert [c.kwargs['headers']['Range'] for c in get.call_args_list] == ['bytes=0-7', 'bytes=16-23']
    assert get.call_args.kwargs['stream'] is True
    assert get.call_args.kwargs['headers']['Authorization'] == 'Bearer test-token'


def test_retry_429_exact_backoff(monkeypatch):
    error = requests.HTTPError(response=Mock(status_code=429))
    op = Mock(side_effect=[error, error, error, 'ok'])
    sleeps = Mock()
    monkeypatch.setattr(hub.time, 'sleep', sleeps)
    assert hub._retry(op) == 'ok'
    assert [c.args[0] for c in sleeps.call_args_list] == [2, 4, 8]
    op = Mock(side_effect=error)
    with pytest.raises(requests.HTTPError):
        hub._retry(op)
    assert op.call_count == 4


TRANSIENT_ERRORS = [
    requests.ConnectionError('connection interrupted'),
    requests.exceptions.ChunkedEncodingError('incomplete response'),
    requests.exceptions.ReadTimeout('read timed out'),
    ProtocolError('raw response interrupted'),
    ReadTimeoutError(None, None, 'raw read timed out'),
    *[requests.HTTPError(response=Mock(status_code=status)) for status in (500, 502, 503, 599)],
]


@pytest.mark.parametrize('error', TRANSIENT_ERRORS)
def test_sample_retries_twice_then_matches(monkeypatch, model_factory, error):
    model = model_factory('model')
    for name in ('resolve_revision', 'list_files', 'read_json', 'read_header'):
        reader = getattr(local, name)
        monkeypatch.setattr(hub, name, lambda repo, *args, reader=reader: reader(str(model), *args))
    payload = (model / 'model.safetensors').read_bytes()
    calls = 0

    def get(url, *, headers, **kwargs):
        nonlocal calls
        calls += 1
        start, end = map(int, headers['Range'].removeprefix('bytes=').split('-'))
        result = response(data=payload[start:end+1], content_range=f'bytes {start}-{end}/{len(payload)}')
        if calls <= 2:
            if isinstance(error, requests.HTTPError):
                result.status_code = error.response.status_code
                result.raise_for_status.side_effect = error
            elif isinstance(error, requests.ConnectionError):
                raise error
            else:
                # Fail in the streamed body, after a valid 206 response.
                result.raw.read.side_effect = error
        return result

    monkeypatch.setattr(hub.requests, 'get', get)
    sleeps = Mock()
    monkeypatch.setattr(hub.time, 'sleep', sleeps)
    report = build_report(str(model), 'owner/repo', max_tensors=1)
    assert len(report.samples) == 1
    assert report.samples[0].status == 'same'
    assert calls == 4  # Two failed reads, then two successful sample windows.
    assert [c.args[0] for c in sleeps.call_args_list] == [2, 4]


@pytest.mark.parametrize('error', TRANSIENT_ERRORS)
def test_transient_retry_stops_after_three_retries(monkeypatch, error):
    operation, sleeps = Mock(side_effect=error), Mock()
    monkeypatch.setattr(hub.time, 'sleep', sleeps)
    with pytest.raises(type(error)):
        hub._retry(operation)
    assert operation.call_count == 4
    assert [c.args[0] for c in sleeps.call_args_list] == [2, 4, 8]


@pytest.mark.parametrize('status', [400, 401, 403, 404, 416])
def test_permanent_http_error_is_not_retried(monkeypatch, status):
    operation = Mock(side_effect=requests.HTTPError(response=Mock(status_code=status)))
    sleeps = Mock()
    monkeypatch.setattr(hub.time, 'sleep', sleeps)
    with pytest.raises(requests.HTTPError):
        hub._retry(operation)
    assert operation.call_count == 1
    sleeps.assert_not_called()


@pytest.mark.parametrize('length', [0, MAX_HEADER + 1, 2**64 - 1])
def test_reject_oversize_header_before_body(tmp_path, length):
    (tmp_path / 'bad.safetensors').write_bytes(struct.pack('<Q', length))
    with pytest.raises(ValueError, match='header length'):
        local.read_header(str(tmp_path), None, 'bad.safetensors')


def test_local_path_escape_and_small_file_allowlist(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        local.read_ranges(str(tmp_path), None, '../outside.bin', [(0, 1)])
    with pytest.raises(ValueError, match='Only named'):
        local.read_json(str(tmp_path), None, 'weights.json')
    assert local.read_json(str(tmp_path), None, 'config.json') is None


def test_truncated_header_is_explicit_collection_note(model_factory):
    model = model_factory('bad')
    (model / 'model.safetensors').write_bytes(struct.pack('<Q', 100) + b'{}')
    result = snapshot(str(model))
    assert not result.tensors
    assert 'Truncated' in result.notes[0]
