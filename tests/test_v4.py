from dataclasses import replace

import pytest

from weightdiff.application.collect import build_report
from weightdiff.domain.flags import make_flags


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('b_vocab', [4096, 8192])
def test_vocabulary_flags_use_config_alias_once(model_factory, reverse, b_vocab):
    root = model_factory('tiny')
    report = build_report(str(root), str(root), sample=False)
    a = replace(report.a, config={'text_config': {'vocab_size': 4096}})
    b = replace(report.b, config={'vocab_size': b_vocab})
    report = replace(report, a=b if reverse else a, b=a if reverse else b)
    flags = [f for f in make_flags(report) if f.title == '語彙の数が違う']
    assert len(flags) == int(b_vocab != 4096)
    if flags:
        va, vb = (b_vocab, 4096) if reverse else (4096, b_vocab)
        assert flags[0].detail == f'text_config.vocab_size ↔ vocab_size: a={va}, b={vb}'
