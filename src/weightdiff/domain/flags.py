import json
from difflib import SequenceMatcher

from .compare import diff_config
from .model import Flag


def _has_key(value, key):
    return isinstance(value, dict) and (key in value or any(_has_key(v, key) for v in value.values()))


def _template(snapshot):
    cfg = snapshot.tokenizer_config or {}
    return {'chat_template': cfg.get('chat_template'), **cfg.get('__weightdiff_chat_templates', {})}


def _lines(value):
    return (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)).splitlines()


def _changed_lines(a, b):
    return sum(max(j-i, l-k) for tag, i, j, k, l in SequenceMatcher(None, _lines(a), _lines(b), autojunk=False).get_opcodes() if tag != 'equal')


def make_flags(report_parts) -> list[Flag]:
    a, b = report_parts.a, report_parts.b
    flags = []
    pickle_files = [f for f in b.files if f.lower().endswith(('.bin', '.pt', '.pth', '.pkl', '.ckpt'))]
    if pickle_files:
        flags.append(Flag('notice', '読み込むとコードが動きうる形式', ', '.join(pickle_files)))
    python_files = [f for f in b.files if f.lower().endswith('.py')]
    if python_files or _has_key(b.config, 'auto_map'):
        flags.append(Flag('notice', '読み込みに trust_remote_code が要る（リポジトリのコードが動く）',
                          ', '.join(python_files) or 'config.json: auto_map'))
    for stack in report_parts.layer_summary:
        if stack.a_count != stack.b_count:
            flags.append(Flag('notice', f'層の数が違う（a: {stack.a_count}、b: {stack.b_count}）', stack.prefix))
    ta, tb = _template(a), _template(b)
    if ta != tb:
        count = sum(_changed_lines(ta.get(k), tb.get(k)) for k in ta.keys() | tb.keys() if ta.get(k) != tb.get(k))
        flags.append(Flag('notice', '会話の書式が違う', f'{count} changed lines (replaced lines counted once)'))
    for change in diff_config(a.config, b.config):
        if change.key in ('vocab_size', 'text_config.vocab_size',
                          'text_config.vocab_size ↔ vocab_size') and change.a_value != change.b_value:
            flags.append(Flag('info', '語彙の数が違う',
                              f'{change.key}: a={change.a_value}, b={change.b_value}'))
    ca = len((a.tokenizer_config or {}).get('added_tokens_decoder') or {})
    cb = len((b.tokenizer_config or {}).get('added_tokens_decoder') or {})
    if ca != cb:
        flags.append(Flag('info', '追加トークンの数が違う', f'added_tokens_decoder: a={ca}, b={cb}'))
    changed = sum(s.status == 'different' for s in report_parts.samples)
    if changed:
        flags.append(Flag('info', f'値が書き換えられたテンソルが {changed} 個（抜き取りの範囲で）', ''))
    if b.format != 'gguf' and not any(f.endswith('.safetensors') for f in b.files):
        flags.append(Flag('info', 'safetensors が無いので、この道具では中身を比べられません', ''))
    return flags
