import os
import tempfile
from pathlib import Path

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import gradio as gr

from .collect import build_report
from .report import (has_quantization, DISCLAIMER, position_notes, flags_markdown, layer_rows, metric_cards,
                     overview_markdown, regions_markdown, read_footer, samples_markdown, tensor_rows, to_json, quantization_markdown)

DASHBOARD_CSS = '''
.gradio-container {
    --wd-different: #B45309;
    --wd-same: #0F766E;
    --wd-text: var(--body-text-color, #111827);
    --wd-muted: #4B5563;
    --wd-surface: var(--body-background-fill, #FFFFFF);
    --wd-border: var(--border-color-primary, #E5E7EB);
    background: var(--wd-surface);
}
.dark .gradio-container, .gradio-container.dark {
    --wd-different: #FBBF24;
    --wd-same: #5EEAD4;
    --wd-muted: #D1D5DB;
}
/* Gradio scopes media rules inside .contain, outside the app wrapper. */
.gradio-container .app { padding: 16px clamp(16px, 2.5vw, 32px) !important; }
#wd-cards .html-container, #wd-footer .html-container { padding: 0; }
.wd-intro-en { font-size: 18px; line-height: 1.5; margin: 0 0 8px; }
.wd-intro-ja { font-size: 15px; line-height: 1.6; color: var(--wd-muted); margin: 0; }
.wd-explanations {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 20px;
    margin-top: 20px;
}
.wd-explanations h2 { font-size: 16px; margin: 0 0 8px; }
.wd-explanations p { font-size: 14px; line-height: 1.6; margin: 0 0 8px; }
.wd-ja, .wd-disclaimer { color: var(--wd-muted); }
.wd-explanations .wd-disclaimer { font-size: 12px; }
.wd-example-caption { font-size: 12px; line-height: 1.5; color: var(--wd-muted); margin: 0; }
.wd-metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}
.wd-card {
    min-width: 0;
    padding: 18px;
    border: 1px solid var(--wd-border);
    border-radius: 12px;
    background: var(--wd-surface);
    color: var(--wd-text);
    display: flex;
    flex-direction: column;
}
.wd-card h2 {
    color: var(--wd-muted);
    font-size: 14px;
    font-weight: 600;
    line-height: 20px;
    margin: 0 0 10px;
}
.wd-number {
    font-size: 38px;
    font-weight: 700;
    line-height: 1.2;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    margin-bottom: 10px;
}
.wd-changed { color: var(--wd-different); }
.wd-unchanged { color: var(--wd-same); }
.wd-card p {
    font-size: 14px;
    line-height: 20px;
    margin: auto 0 0;
    color: var(--wd-muted);
    white-space: normal;
    overflow-wrap: anywhere;
}
#wd-overview li { margin-bottom: 6px; }
#wd-footer { color: var(--wd-muted); font-size: 12px; }
@media (max-width: 640px) {
    .wd-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .wd-card { padding: 12px; }
    .wd-number { font-size: 28px; }
    .wd-number-long { font-size: 24px; }
    .wd-explanations { grid-template-columns: 1fr; }
}
'''

INTRO_HTML = f'''
<p class="wd-intro-en">Check a model against the base model it claims to come from — without downloading the weights.</p>
<p class="wd-intro-ja" lang="ja">“〇〇をもとにした” と書かれたモデルに、元のモデルから変な改造がされていないかを、重みをダウンロードせずに確かめる道具です。</p>
<div class="wd-explanations">
    <section>
        <h2>What it reads <span class="wd-ja" lang="ja">/ 読むもの</span></h2>
        <p>Only file headers and small samples per tensor (a few MB to a few hundred MB for a 4B model; never the full weights).</p>
        <p class="wd-ja" lang="ja">ファイルの頭と、数字の表ごとの小さな抜き取りだけ（4B のモデルで数 MB〜数百 MB。重み全体は読みません）。</p>
    </section>
    <section>
        <h2>What it shows <span class="wd-ja" lang="ja">/ 分かること</span></h2>
        <p>Added or removed layers, widened shapes, changed values, extra code or pickle files, chat-template changes.</p>
        <p class="wd-ja" lang="ja">足された・消えた層、広げられた形、書き換えられた値、余分なコードや pickle、会話の書式の変更。</p>
    </section>
    <section>
        <h2>What it can't do <span class="wd-ja" lang="ja">/ 分からないこと</span></h2>
        <p>It cannot prove a model is safe. Small value changes can slip past sampling.</p>
        <p class="wd-ja" lang="ja">安全の証明はできません。小さな値の書き換えは抜き取りで見逃すことがあります。</p>
        <p class="wd-disclaimer">{DISCLAIMER}</p>
    </section>
</div>
'''

EXAMPLE_CAPTIONS = [
    'Same model twice — nothing differs.',
    'A normal fine-tune — same structure, many values retrained.',
    'Same family name, but widened 2560 → 3968 and 11 layers added.',
    'A Q8_0 GGUF of the same model — compare with quantization noise in mind.',
    'A Q4_K_M GGUF of the same model — compare with quantization noise in mind.',
    'A Q8_0 GGUF of the fine-tuned model — compare sampled values after dequantization.',
]

EXAMPLES = [
    ('Identical', 'Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-4B'),
    ('Fine-tuned', 'Qwen/Qwen3.5-4B', 'ccore/Qwen3.5-4B-3D-Coder'),
    ('Widened + extra layers', 'Qwen/Qwen3.5-4B', 'divagr1925/Qwen3.5-4B-Qwenputer'),
    ('GGUF Q8_0', 'Qwen/Qwen3.5-4B', 'unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q8_0.gguf'),
    ('GGUF Q4_K_M', 'Qwen/Qwen3.5-4B', 'unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf'),
    ('Fine-tuned GGUF', 'Qwen/Qwen3.5-4B', 'ccore/Qwen3.5-4B-3D-Coder:Qwen3.5-4B-3D-Coder-Q8_0.gguf'),
]


def compare(a, b, rev_a, rev_b, sample, progress=gr.Progress()):
    if not a.strip() or not b.strip():
        raise gr.Error('Enter both model references.')
    try:
        report = build_report(a.strip(), b.strip(), (rev_a or '').strip() or None, (rev_b or '').strip() or None,
                              sample=sample, progress=progress)
    except Exception as exc:
        raise gr.Error(f'Comparison could not complete: {exc}') from exc
    output = Path(os.environ.get('WEIGHTDIFF_OUTPUT_DIR', Path(__file__).resolve().parents[3] / 'out'))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='weightdiff-', dir=output,
                                     delete=False, encoding='utf-8') as file:
        file.write(to_json(report))
        download = file.name
    files = ([['Only B', name] for name in report.file_diff.only_b]
             + [['Only A', name] for name in report.file_diff.only_a])
    d = report.tensor_diff
    tensor_note = (f'{d.excluded_by_layer_shift} tensor names excluded from name comparison because layer counts differ. '
                   f'Parameters: A {d.a_parameters:,} · B {d.b_parameters:,}. '
                   'Showing the first 50 differences; download JSON for all.')
    return (metric_cards(report), overview_markdown(report) + ('\n\n' + quantization_markdown(report) if has_quantization(report) else ''), flags_markdown(report),
            layer_rows(report, changed_limit=8), position_notes(report), tensor_rows(report)[:50], tensor_note,
            [[c.key, str(c.a_value), str(c.b_value), c.kind] for c in report.config_diff],
            files, samples_markdown(report), regions_markdown(report), download, read_footer(report))


with gr.Blocks(title='weightdiff', analytics_enabled=False) as demo:
    gr.Markdown('# weightdiff')
    gr.HTML(INTRO_HTML, elem_id='wd-intro')
    with gr.Row():
        with gr.Column(min_width=250):
            a = gr.Textbox(label='Model A', value='Qwen/Qwen3.5-4B')
        with gr.Column(min_width=250):
            b = gr.Textbox(label='Model B', placeholder='Hub model ID, repo:file.gguf, local folder or .gguf path')
    with gr.Accordion('Revisions (optional)', open=False):
        with gr.Row():
            rev_a = gr.Textbox(label='Revision A')
            rev_b = gr.Textbox(label='Revision B')
    with gr.Row():
        sample = gr.Checkbox(label='Also compare sampled values', value=True, scale=2)
        run = gr.Button('Compare', variant='primary', scale=1)
    gr.Markdown('Examples — click to run a comparison:')
    examples = []
    for offset in (0, 3):
        with gr.Row():
            for (label, left, right), caption in zip(EXAMPLES[offset:offset + 3], EXAMPLE_CAPTIONS[offset:offset + 3]):
                with gr.Column(min_width=200):
                    examples.append((gr.Button(label), left, right))
                    gr.HTML(f'<p class="wd-example-caption">{caption}</p>')
    cards = gr.HTML(elem_id='wd-cards')
    summary = gr.Markdown(elem_id='wd-overview')
    gr.Markdown('## Points to note')
    flags = gr.Markdown(elem_id='wd-flags')
    with gr.Accordion('Layers', open=False):
        layers = gr.Dataframe(label='Layers', headers=['Stack', 'A', 'B', 'Added', 'Removed', 'Added positions (B)', 'Removed positions (A)', 'Changed (A, B)', 'Fingerprints (A / B)', 'Pairing', 'Position basis'], interactive=False)
        positions = gr.Markdown()
    with gr.Accordion('Tensor differences', open=False):
        tensor_note = gr.Markdown()
        tensors = gr.Dataframe(label='Tensor differences', headers=['Change', 'Tensor', 'A shape', 'A dtype', 'B shape', 'B dtype'], interactive=False)
    with gr.Accordion('Configuration differences', open=False):
        config = gr.Dataframe(label='Configuration differences', headers=['Key', 'A', 'B', 'Change'], interactive=False)
    with gr.Accordion('Files', open=False):
        files = gr.Dataframe(label='File differences', headers=['Change', 'File'], interactive=False)
    with gr.Accordion('Sampled values', open=False):
        samples = gr.Markdown()
    with gr.Accordion('New regions in B / B で新しく増えた部分', open=False):
        regions = gr.Markdown()
    download = gr.File(label='Download JSON', interactive=False)
    footer = gr.HTML(elem_id='wd-footer')
    outputs = [cards, summary, flags, layers, positions, tensors, tensor_note, config, files, samples, regions, download, footer]
    run.click(compare, [a, b, rev_a, rev_b, sample], outputs, concurrency_limit=1)
    for button, left, right in examples:
        button.click(lambda left=left, right=right: (left, right, '', ''), outputs=[a, b, rev_a, rev_b], queue=False).then(
            compare, [a, b, rev_a, rev_b, sample], outputs, concurrency_limit=1)

demo.queue(default_concurrency_limit=1)



def main():
    demo.launch(server_name="127.0.0.1", css=DASHBOARD_CSS)


if __name__ == '__main__':
    main()
