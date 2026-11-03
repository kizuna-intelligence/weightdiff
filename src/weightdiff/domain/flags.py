import json
from difflib import SequenceMatcher

from .compare import diff_config
from .layers import layer_parts
from .model import Flag


def _has_key(value, key):
    return isinstance(value, dict) and (key in value or any(_has_key(v, key) for v in value.values()))


def _template(snapshot):
    cfg = snapshot.tokenizer_config or {}
    files = cfg.get('__weightdiff_chat_templates', {})
    # ProcessorMixin.get_processor_dict reads legacy JSON before raw Jinja;
    # PreTrainedTokenizerBase._from_pretrained prefers Jinja over tokenizer config.
    if files.get('chat_template.json') is not None:
        legacy = files['chat_template.json']
        return legacy.get('chat_template', legacy)
    if files.get('chat_template.jinja') is not None:
        return files['chat_template.jinja']
    return cfg.get('chat_template')


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
        count = _changed_lines(ta,tb)
        flags.append(Flag('notice', '会話の書式が違う', f'{count} changed lines (replaced lines counted once)'))
    elif ta is not None and (a.tokenizer_config or {}) != (b.tokenizer_config or {}):
        def locations(snap):
            cfg = snap.tokenizer_config or {}
            return (cfg.get('chat_template'), cfg.get('__weightdiff_chat_templates', {}))
        if locations(a) != locations(b):
            flags.append(Flag('info', '会話の書式は同じ（置き場所だけ違う）', 'Effective chat templates match.'))
    for change in diff_config(a.config, b.config):
        if change.key in ('vocab_size', 'text_config.vocab_size',
                          'text_config.vocab_size ↔ vocab_size') and change.a_value != change.b_value:
            flags.append(Flag('info', '語彙の数が違う',
                              f'{change.key}: a={change.a_value}, b={change.b_value}'))
    ca = len((a.tokenizer_config or {}).get('added_tokens_decoder') or {})
    cb = len((b.tokenizer_config or {}).get('added_tokens_decoder') or {})
    if ca != cb:
        flags.append(Flag('info', '追加トークンの数が違う', f'added_tokens_decoder: a={ca}, b={cb}'))
    changed = sum(s.status in ('different', 'rescaled') for s in report_parts.samples)
    if changed:
        flags.append(Flag('info', f'値が書き換えられたテンソルが {changed} 個（抜き取りの範囲で）', ''))
    if b.format != 'gguf' and not any(f.endswith('.safetensors') for f in b.files):
        flags.append(Flag('info', 'safetensors が無いので、この道具では中身を比べられません', ''))
    baseline = max((s.max_abs_a or 0 for s in report_parts.samples),default=0.)
    threshold = max(1e4,10*baseline)
    def region_label(name, region):
        parts = layer_parts(name)
        layer = f'B layer {parts[1]} ({parts[0]})' if parts else 'B outside layers'
        return f'{name} [{region}; {layer}]'
    extreme = [(s.name.rsplit(' → ',1)[-1], 'paired_sample', s.max_abs_b)
               for s in report_parts.samples if s.max_abs_b is not None and s.max_abs_b>threshold]
    extreme += [(r.name,r.region,r.max_abs) for r in report_parts.new_regions if r.max_abs>threshold]
    if extreme:
        examples = sorted(set(extreme),key=lambda x:(-x[2],x[0],x[1]))[:5]
        flags.append(Flag('notice','普通の重みではありえない大きさの値',
                          f'|x| > {threshold:.6g}: '+ '; '.join(f'{region_label(n,r)}: {v:.6g}' for n,r,v in examples)))
    sparse = [r for r in report_parts.new_regions if r.region == 'new_tensor'
              and len(b.tensors[r.name].shape)==2 and r.count>=1024 and r.zero_fraction>=.99]
    if sparse:
        flags.append(Flag('notice','ほとんど 0 の新しい重み（学習した重みには見えない。手で置いたか、0 で初期化した物）',
                          f'{len(sparse)} regions; '+', '.join(region_label(r.name,r.region) for r in sparse[:3])))
    extra = [r for r in report_parts.new_regions if r.region.startswith('extra_')]
    used = sorted((r for r in extra if r.zero_fraction<1),key=lambda r:(-r.max_abs,r.name,r.region))
    zero = [r for r in extra if r.zero_fraction==1]
    if used:
        flags.append(Flag('info',f'広げた部分が使われている（{len(used)} 領域）',
                          '; '.join(f'{region_label(r.name,r.region)}: {r.max_abs:.6g}' for r in used[:5])))
    if zero:
        flags.append(Flag('info',f'広げた部分は 0 埋め（{len(zero)} 領域）',
                          f'{len(zero)} sampled widened regions are zero.'))
    return flags
