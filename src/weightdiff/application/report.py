import json
from collections import Counter
from dataclasses import asdict
from html import escape

from ..domain.compare import sample_pairs
from ..domain.layers import layer_parts

DISCLAIMER = 'この道具は構造の違いを見せるだけで、モデルが安全であることは証明できません。/ This tool shows structural differences. It cannot prove that a model is safe.'
UNKNOWN_POSITION = 'Added layers have the same shape as existing layers, so their positions cannot be inferred from shapes. / 足された層は既存の層と同じ形のため、位置は形からは分かりません'
UNKNOWN_NAME_POSITION = 'Added layers have existing name fingerprints, so their positions cannot be inferred from names.'


def position_notes(report):
    return '\n\n'.join(text for basis, text in [('shape', UNKNOWN_POSITION), ('name', UNKNOWN_NAME_POSITION)]
                         if any(not s.position_known and (s.pairing_basis == basis or (basis == 'shape' and s.pairing_basis == 'width')) for s in report.layer_summary))


def _cell(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    return escape(text).replace('|', '&#124;').replace('\n', '<br>')


def _table(headers, rows, limit=50):
    rows = list(rows)
    if not rows:
        return 'No differences.\n'
    result = ['| ' + ' | '.join(map(_cell, headers)) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    result += ['| ' + ' | '.join(map(_cell, row)) + ' |' for row in rows[:limit]]
    if len(rows) > limit:
        result.append(f'\n{len(rows)-limit:,} more rows. Full details are in JSON.')
    return '\n'.join(result) + '\n'


def tensor_rows(report):
    d = report.tensor_diff
    paired = dict(d.name_pairs)
    rows = ([['Name location changed', f'{d.renamed_count} tensors', '', '', '', '']]
            if d.renamed_count else [])
    if report.a.format != report.b.format:
        rows += [['No correspondence', f'A: {len(report.a.unmapped_names)} · B: {len(report.b.unmapped_names)}', '', '', '', '']]
    for kind, names in [('Only A', d.only_a), ('Only B', d.only_b), ('Shape changed', d.shape_changed), ('Dtype changed', d.dtype_changed)]:
        for name in names:
            a = report.a.tensors.get(name) if kind != 'Only B' else None
            b = report.b.tensors.get(paired.get(name, name)) if kind != 'Only A' else None
            rows.append([kind, name, str(a.shape) if a else '', a.dtype if a else '', str(b.shape) if b else '', b.dtype if b else ''])
    return rows


def layer_rows(report, changed_limit=None):
    rows = []
    for layer in report.layer_summary:
        fingerprints = '; '.join(f'F{i+1}: {f.a_count} / {f.b_count}' for i, f in enumerate(layer.fingerprints))
        changed = str(list(layer.changed if changed_limit is None else layer.changed[:changed_limit]))
        if changed_limit is not None and len(layer.changed) > changed_limit:
            changed += f' … {len(layer.changed) - changed_limit} more'
        rows.append([layer.prefix, layer.a_count, layer.b_count, layer.added_count, layer.removed_count,
                     str(list(layer.added)) if layer.position_known else f'Unknown from {layer.pairing_basis}s',
                     str(list(layer.removed)), changed, fingerprints,
                     ('Name correspondence (GGUF); dtype/shape ignored for alignment'
                      if report.a.format == 'gguf' or report.b.format == 'gguf' else
                      'Paired by names (shapes differ) / 名前で組（形が違う）')
                     if layer.pairing_basis == 'name' else ('Width-normalized fingerprints (suffix and shape; dtype ignored)' if layer.pairing_basis == 'width' else 'Shape fingerprints'),
                     f'{layer.position_basis}: ' + ('値で確定 / confirmed by sampled values' if layer.position_basis == 'samples' else '形だけ、あいまい / structure only; repeated shapes are ambiguous')])
    return rows


def _positions(numbers):
    if not numbers:
        return 'none'
    if len(numbers) > 1 and tuple(numbers) == tuple(range(numbers[0], numbers[-1] + 1)):
        return f'{numbers[0]}–{numbers[-1]}'
    return ', '.join(map(str, numbers))


def stack_sample_counts(report):
    paired_layers = {}
    for na, nb in sample_pairs(report.a, report.b, report.tensor_diff, report.layer_summary):
        pa, pb = layer_parts(na), layer_parts(nb)
        for stack in report.layer_summary:
            if pa and pb and pa[0] == stack.a_prefix and pb[0] == stack.b_prefix:
                paired_layers.setdefault(stack.prefix, set()).add((pa[1], pb[1]))
    return [(s, len(paired_layers.get(s.prefix, ())),
             Counter(x.status for x in report.samples if x.stack_prefix == s.prefix))
            for s in report.layer_summary]


def has_quantization(report):
    return (report.a.format == 'gguf' or report.b.format == 'gguf'
            or any(t.quantization for snap in (report.a,report.b) for t in snap.tensors.values()))


def quantization_markdown(report):
    lines = []
    for noise in report.quant_noise:
        source = 'expected error from A roundtrip' if noise.source == 'A' else 'observed error in B (fallback)'
        if noise.source == 'A' and noise.median == 0 and noise.dtype.startswith(('bnb_', 'gptq', 'awq')):
            lines.append(f'- **{noise.dtype}**: zero measured RTN error on sampled values; {noise.count} tensors. '
                         'The quantized format is retained. / 抜き取りでの RTN 誤差は 0（量子化形式）')
            continue
        if noise.source == 'A' and noise.median == 0:
            lines.append(f'- **{noise.dtype}**: no quantization; {noise.count} tensors compared exactly '
                         '(float32 rounding only). / 量子化なし。丸めの範囲で厳密に比較')
            continue
        lines.append(f'- **{noise.dtype}**: {source}; median {noise.median:.8g} '
                     f'({noise.median * 100:.4g}%), MAD {noise.mad:.8g}, {noise.count} tensors. '
                     f'For this dtype, changes smaller than about {noise.undetectable_percent:.4g}% of the original values '
                     'cannot be distinguished. / この型では、元の値の約 '
                     f'{noise.undetectable_percent:.4g}% より小さい書き換えは見分けられません')
        if noise.source == 'B':
            lines.append(f'  **{noise.dtype}: Noise floor estimated from B itself; uniformly retrained values '
                         'cannot be distinguished. / この型の雑音の床は B 自身から推定'
                         '（全体を一様に変えた学習は見分けられません）** '
                         + _cell(noise.detail))
    return '\n'.join(lines) or 'No quantization noise estimate is available.'


def quantization_counts(report, *, card=False):
    counts = Counter(s.status for s in report.samples)
    differences = f'beyond {counts["beyond_quant_noise"]} · uniform {counts["systematic"]}'
    if card:
        return differences
    text = (f'within {counts["within_quant_noise"]} · requant-identical {counts["same_after_requant"]} '
            f'· transform {counts["same_after_transform"]} · rearranged {counts["rearranged"]}')
    return text + ' · ' + differences


def overview_sentences(report):
    parts = []
    for stack, paired, counts in stack_sample_counts(report):
        name = stack.b_prefix or stack.a_prefix
        basis = 'name' if stack.pairing_basis == 'name' else 'shape'
        if stack.position_basis == 'samples': basis = 'samples'
        if stack.b_count > stack.a_count:
            positions = (f'positions {_positions(stack.added)}, confirmed by samples' if stack.position_basis == 'samples' else f'positions {_positions(stack.added)}, new layer {basis}'
                         if stack.position_known else f'positions unknown from {basis}s')
            parts.append(f'B has {stack.b_count - stack.a_count} more layers than A in {name} '
                         f'(A {stack.a_count}, B {stack.b_count}; {positions}).')
        elif stack.b_count < stack.a_count:
            parts.append(f'B has {stack.a_count - stack.b_count} fewer layers than A in {name} '
                         f'(A {stack.a_count}, B {stack.b_count}; removed positions {_positions(stack.removed) if stack.position_known else "unknown"}, {stack.position_basis}).')
        else:
            parts.append(f'A and B each have {stack.a_count} layers in {name}.')
        if paired and (has_quantization(report)):
            parts.append(f'{paired} layers pair with A\'s layers; quantized samples: '
                         f'{counts["same_after_requant"]} same after requant, '
                         f'{counts["within_quant_noise"]} within noise, {counts["beyond_quant_noise"]} beyond, '
                         f'{counts["same_after_transform"]} transform, {counts["rearranged"]} rearranged, '
                         f'{counts["systematic"]} uniform, {counts["same"]} byte-identical, '
                         f'{counts["different"]} byte-different, {counts["skipped"]} skipped.')
        elif paired:
            parts.append(f'{paired} layers pair with A\'s layers; sampled values: '
                         f'{counts["same"]} same, {counts["different"] + counts["rescaled"]} different (rescaled {counts["rescaled"]}), {counts["skipped"]} skipped.')
            if counts['embedded_top_left'] or counts['not_top_left']:
                parts.append(f'Paired-layer top-left samples: {counts["embedded_top_left"]} embedded_top_left, '
                             f'{counts["not_top_left"]} not_top_left.')
    if report.tensor_diff.axis_changes:
        change = report.tensor_diff.axis_changes[0]
        parts.append(f'Width {change.a_size} → {change.b_size} in {change.tensors} tensors.')
    samples = Counter(s.status for s in report.samples)
    top_left_total = samples['embedded_top_left'] + samples['not_top_left']
    if top_left_total:
        parts.append('Sampled A tensors found as the top-left block of the wider B tensor: '
                     f'{samples["embedded_top_left"]} of {top_left_total}.')
    parts.append(f'Total sampled tensors: {len(report.samples)} of {report.sample_candidates}; '
                 f'{samples["same"]} same, {samples["different"] + samples["rescaled"]} different (rescaled {samples["rescaled"]}), {samples["skipped"]} skipped'
                 + (f', {samples["embedded_top_left"]} embedded_top_left, {samples["not_top_left"]} not_top_left.'
                    if top_left_total else '.')
                 if report.sampling_enabled else 'Sampling disabled; no values were compared.')
    if has_quantization(report):
        beyond, uniform = samples['beyond_quant_noise'], samples['systematic']
        if report.sampling_enabled:
            parts.append(f'{beyond + uniform} tensors differ beyond quantization noise '
                         f'({beyond} isolated, {uniform} uniform across sampled layers).')
        parts.append('Quantized samples: ' + quantization_counts(report) + '.')
        parts.append(f'Name correspondence unavailable: A {len(report.a.unmapped_names)}, B {len(report.b.unmapped_names)} tensors.')
    if report.new_regions:
        used = sum(r.region.startswith('extra_') and r.zero_fraction<1 for r in report.new_regions)
        notices = [f for f in report.flags if f.title.startswith(('普通の重み','ほとんど 0'))]
        parts.append(f'New B regions sampled: {len(report.new_regions)}; widened regions in use: {used}; '
                     f'new-region notices: {len(notices)}.')
    return parts


def overview_sentence(report):
    return ' '.join(overview_sentences(report))


def overview_markdown(report):
    return '\n'.join(f'- {_cell(sentence)}' for sentence in overview_sentences(report))


def metric_cards(report):
    """Present collected results without changing any comparison rules."""
    stack = max(report.layer_summary, key=lambda s: max(s.a_count, s.b_count), default=None)
    if stack:
        delta = stack.b_count - stack.a_count
        layers = f'{stack.a_count} → {stack.b_count}'
        layer_detail = f'{delta:+d} layers' if delta else 'same count'
    else:
        delta, layers, layer_detail = 0, '—', 'unavailable'

    def width(snapshot):
        config = snapshot.config or {}
        text = config.get('text_config')
        nested = text.get('hidden_size') if isinstance(text, dict) else None
        return nested if nested is not None else config.get('hidden_size')

    wa, wb = width(report.a), width(report.b)
    width_changed = wa != wb
    width_value = str(wa) if wa == wb and wa is not None else f'{wa if wa is not None else "—"} → {wb if wb is not None else "—"}'
    width_detail = 'hidden size changed' if width_changed else ('unchanged' if wa is not None else 'unavailable')
    counts = Counter(s.status for s in report.samples)
    differences = counts["different"] + counts["rescaled"]
    sample_detail = f'same {counts["same"]} · different {differences} · rescaled {counts["rescaled"]}'
    wider = counts['embedded_top_left'] + counts['not_top_left']
    if wider:
        sample_title = 'Original values inside'
        sample_value = f'{counts["embedded_top_left"]} / {wider}'
        sample_detail = f'found in the top-left of wider tensors · different {differences} · rescaled {counts["rescaled"]}'
    else:
        sample_title = 'Sampled values'
        sample_value = str(differences)
    if has_quantization(report):
        sample_title = 'Quantized values'
        sample_value = str(counts['beyond_quant_noise'] + counts['systematic'])
        sample_detail = quantization_counts(report, card=True)
    if not report.sampling_enabled:
        sample_detail = 'Sampling disabled'
    notices = [f for f in report.flags if f.level == 'notice']
    cards = [
        ('Layers', layers, layer_detail, bool(delta)),
        ('Width', width_value, width_detail, width_changed),
        (sample_title, sample_value, sample_detail,
         differences > 0 or counts['beyond_quant_noise'] + counts['systematic'] > 0),
        ('Points to note', str(len(notices)), _flag_title(notices[0]) if notices else 'No notices', bool(notices)),
    ]
    html = ['<div class="wd-metrics" aria-label="Comparison summary">']
    for title, value, detail, changed in cards:
        number = escape(value)
        size_class = ' wd-number-long' if len(value) > 9 else ''
        html.append(f'<section class="wd-card"><h2>{escape(title)}</h2>'
                    f'<div class="wd-number {"wd-changed" if changed else "wd-unchanged"}{size_class}">{number}</div>'
                    f'<p title="{escape(detail, quote=True)}">{escape(detail)}</p></section>')
    return ''.join(html) + '</div>'


def read_footer(report):
    return (f'<small>{report.read_stats.get("total_bytes", 0):,} bytes read · '
            f'{report.elapsed_seconds:.2f} seconds. Full weight files are never downloaded.</small>')


def summary_markdown(report):
    d = report.tensor_diff
    changed = len(set(d.only_a + d.only_b + d.shape_changed + d.dtype_changed))
    samples = Counter(s.status for s in report.samples)
    layers = '; '.join(f'{s.prefix}: {s.a_count} → {s.b_count}' for s in report.layer_summary) or 'unavailable'
    lines = [
        f'- **Parameters:** A {d.a_parameters:,} · B {d.b_parameters:,}',
        f'- **Layers:** {_cell(layers)}',
        f'- **Differences:** {len(report.config_diff)} config keys · {changed} tensor names · {d.renamed_count} name locations changed · {len(report.file_diff.only_a)+len(report.file_diff.only_b)} file paths',
        f'- **Samples:** {len(report.samples)} of {report.sample_candidates} tensors · {samples["same"]} same · {samples["different"] + samples["rescaled"]} different · rescaled {samples["rescaled"]} · {samples["skipped"]} skipped · {samples["embedded_top_left"]} embedded_top_left · {samples["not_top_left"]} not_top_left',
        f'- **Read:** {report.read_stats.get("total_bytes",0):,} bytes · {report.elapsed_seconds:.2f} seconds '
        f'({report.read_stats.get("header_bytes",0):,} header + {report.read_stats.get("sample_bytes",0):,} sample + {report.read_stats.get("small_file_bytes",0):,} small-file + {report.read_stats.get("region_bytes",0):,} new-region + {report.read_stats.get("position_bytes",0):,} position-probe bytes)',
    ]
    lines += [f'- **{_cell(s.prefix)}:** {paired} layers paired with A; sampled values: '
              f'{counts["same"]} same / {counts["different"] + counts["rescaled"]} different / rescaled {counts["rescaled"]} / {counts["skipped"]} skipped / '
              f'{counts["embedded_top_left"]} embedded_top_left / {counts["not_top_left"]} not_top_left'
              for s, paired, counts in stack_sample_counts(report)]
    if has_quantization(report):
        lines.append('- **Quantized samples:** ' + quantization_counts(report))
        lines.append(f'- **No name correspondence:** A {len(report.a.unmapped_names)} · B {len(report.b.unmapped_names)}')
        lines.append(quantization_markdown(report))
    lines += [f'- **Axis size:** {c.a_size} → {c.b_size}: {c.tensors} tensors'
              for c in d.axis_changes]
    return overview_sentence(report) + '\n\n' + '\n'.join(lines)


def _flag_title(flag):
    title = flag.title
    translations = {
        '読み込むとコードが動きうる形式': 'A file format that can execute code when loaded',
        '読み込みに trust_remote_code が要る（リポジトリのコードが動く）': 'Loading requires trust_remote_code (repository code executes)',
        '会話の書式が違う': 'Chat formatting differs',
        '会話の書式は同じ（置き場所だけ違う）': 'Chat formatting is the same (storage location differs)',
        '普通の重みではありえない大きさの値': 'Values exceed ordinary weight magnitudes',
        'ほとんど 0 の新しい重み（学習した重みには見えない。手で置いたか、0 で初期化した物）': 'New weights are almost zero (may be hand-set or zero-initialized)',
        '語彙の数が違う': 'Vocabulary sizes differ',
        '追加トークンの数が違う': 'Added-token counts differ',
        'safetensors が無いので、この道具では中身を比べられません': 'No safetensors: this tool cannot compare the contents',
    }
    if title in translations:
        return translations[title]
    for prefix, english in [('広げた部分が使われている', 'Widened regions are in use'),
                            ('広げた部分は 0 埋め', 'Sampled widened regions are zero-filled')]:
        if title.startswith(prefix):
            count = title[len(prefix):].removeprefix('（').removesuffix(' 領域）')
            return f'{english} ({count} regions)'
    if title.startswith('層の数が違う'):
        return title.replace('層の数が違う', 'Layer counts differ').replace('（', ' (').replace('）', ')').replace('、', ', ')
    if title.startswith('値が書き換えられたテンソルが '):
        count = title.split()[1]
        return f'Values changed in {count} tensors (within sampled ranges)'
    return title


def flags_markdown(report):
    flags = sorted(report.flags, key=lambda f: f.level != 'notice')
    lines = [f'- **{f.level}: {_cell(_flag_title(f))}** {_cell(f.detail)}' for f in flags]
    for label, snapshot in [('A', report.a), ('B', report.b)]:
        lines.extend(f'- **Collection note ({label}):** {_cell(note)}' for note in snapshot.notes)
    return '\n'.join(lines) or 'No points to note were found in the collected metadata.'


def samples_markdown(report):
    total = report.sample_candidates
    count = len(report.samples)
    if not report.sampling_enabled:
        return 'Sampling disabled. No values were compared.'
    if not total:
        return 'No tensors have matching names (after pairing) with matching or wider shapes. No sampled values could be compared.'
    if not count:
        return f'{total} candidate tensors; sampling limit is zero. No values were compared.'
    description = f'{count} of {total} candidate tensors sampled.'
    if count < total:
        description += ' Names were evenly subsampled.'
    if any(t.quantization for snap in (report.a,report.b) for t in snap.tensors.values()):
        description += ('\n\nQuantized safetensors use logical shapes and bounded packed blocks, scales and zeros. '
            'bitsandbytes compares reconstructed values with A quantized using the same code map and block size '
            '(same_after_requant / 量子化だけの違い). GPTQ/AWQ use 1.5 × the error of A rounded to the same '
            'bits and groups as their noise floor; g_idx selects act_order groups. '
            'Wider tensors compare the top-left rows. Unavailable codecs or incomplete reference blocks '
            'are skipped with a reason. Unread values have not been compared.\n\n')
        description += quantization_markdown(report) + '\n\n'
        return description + _table(['Tensor','Result','Relative error','Threshold','Detail'],
            [(s.name,s.status,s.relative_error,s.threshold,s.reason) for s in report.samples])
    if has_quantization(report):
        description += ('\n\nSamples align to quantization blocks (first and middle 2,048 elements). '
            'Same-typed GGUF pairs compare exact block bytes. Other pairs are dequantized to float32. '
            'same_after_requant: the original quantized to B’s dtype matches exactly. '
            'Expected noise e_exp comes from quantizing and restoring A’s samples to B’s dtype. '
            'The threshold is 1e-5 when e_exp = 0, otherwise max(2 × e_exp, e_exp + 0.005). '
            'Unsupported encoders fall back to B’s '
            'per-dtype median/MAD, explicitly marked below. Different GGUF dtypes use the larger floor. '
            'Above-threshold samples are checked for element transforms (−x, x+1, x−1, −exp(x), exp(x), log(x) '
            'on positive A values), then one permutation of intact head-sized column groups inferred '
            'by minimum unsorted relative error on the first complete row, with no reused A groups, '
            'and applied unchanged to the middle and last rows. Group sizes come from config/GGUF '
            'head dimensions, or 64/128/256 when absent. Row permutations compare unsorted rows '
            'against distinct A rows and may additionally read A tensors of at most 32 MiB. '
            'Each transform is also checked in combination with rearrangement. Only 1D samples compare '
            'sorted values, at a threshold of at most 1e-5. Matches are same_after_transform or rearranged; '
            'other samples are beyond_quant_noise. '
            'Three or more sampled layers of the same name kind, all beyond with error MAD/median < 0.2, '
            'are systematic: uniform differences across sampled layers (retraining or an unknown transform). '
            'Both beyond_quant_noise and systematic count as differences. Unsampled layers are not inferred. '
            'These checks do not prove unchanged weights.\n\n')
        description += quantization_markdown(report) + '\n\n'
        description += _table(['Tensor', 'Result', 'Relative error', 'Expected errors (A)', 'Transform', 'Detail'],
            [(s.name, s.status, s.relative_error, s.expected_errors, s.transform, s.reason) for s in report.samples], limit=50)
        return description
    description += '\n\nEach sample uses the first 2,048 elements and, for tensors with more than 8,192 elements, the middle 2,048 elements. '
    description += 'Same dtype: same = the sampled bytes are identical. '
    description += 'Different floating dtypes: same = sampled values are the same (within dtype-conversion rounding). '
    description += 'Different = values differ (have been rewritten). Unread ranges are skipped. '
    description += 'Identical samples do not establish identical weights.\n\n'
    description += ('For wider B tensors: 2D samples compare complete A-width rows at row 0, the middle row, '
                    'and the last A row, against the same row numbers in B. 1D samples compare all A elements '
                    'up to 8,192 elements, otherwise the first and middle 2,048. 3D and higher are skipped. '
                    'embedded_top_left means these sampled values matched; not_top_left means not found in the '
                    'top-left block (other placements have not been checked).\n\n')
    nonmatching = [(s.name, s.status, s.reason) for s in report.samples if s.status != 'same']
    description += 'rescaled is counted as a difference; the sampled 1D values fit a uniform scale (direct or 1+w).\n\n'
    description += (_table(['Tensor', 'Result', 'Detail'], nonmatching) if nonmatching
                    else 'All sampled tensors matched under the comparison rules above.')
    return description


def to_markdown(report):
    d = report.tensor_diff
    parts = [DISCLAIMER, '## Overview', summary_markdown(report),
             f'A: {_cell(report.a.ref)} @ {_cell(report.a.revision)}  \nB: {_cell(report.b.ref)} @ {_cell(report.b.revision)}',
             'Read bytes count file payloads actually inspected, including cached small files; HTTP headers and Hub API listings are excluded. '
             'Only headers and bounded samples of weight files are read; full weight files are never downloaded.',
             '## Points to note', flags_markdown(report), '## Layers',
             _table(['Stack', 'A', 'B', 'Added', 'Removed', 'Added positions (B)', 'Removed positions (A)', 'Changed (A, B)', 'Fingerprints (A / B)', 'Pairing', 'Position basis'], layer_rows(report))]
    if position_notes(report):
        parts.append(position_notes(report))
    parts += ['## Configuration', _table(['Key', 'A', 'B', 'Change'], [(c.key, c.a_value, c.b_value, c.kind) for c in report.config_diff]),
              '## Tensors', f'{d.excluded_by_layer_shift} tensor names excluded because their layer stack changed count. '
              f'Stored tensor bytes: A {d.a_bytes:,}; B {d.b_bytes:,}.',
              _table(['Change', 'Tensor', 'A shape', 'A dtype', 'B shape', 'B dtype'], tensor_rows(report)),
              '## Samples', samples_markdown(report), '## B で新しく増えた部分 / New regions in B', regions_markdown(report), '## Files',
              _table(['Change', 'File'], [('Only B', f) for f in report.file_diff.only_b] + [('Only A', f) for f in report.file_diff.only_a])]
    return '\n\n'.join(parts) + '\n'


def to_json(report):
    return json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True) + '\n'


def regions_markdown(report):
    if not report.sampling_enabled: return 'Sampling disabled. No new regions were read.'
    rows = [(r.name,r.region,r.count,f'{r.zero_fraction:.6g}',f'{r.max_abs:.8g}',r.finite)
            for r in sorted(report.new_regions,key=lambda r:(-r.max_abs,r.name,r.region))]
    return 'Bounded samples only; zeros and maxima describe the inspected elements.\n\n' + _table(
        ['Tensor','Region','Count','Zero fraction','Max absolute value','All finite'],rows)
