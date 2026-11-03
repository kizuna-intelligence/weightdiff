# weightdiff

“〇〇をもとにした” と書かれたモデルに、元のモデルから変な改造がされていないかを、重みをダウンロードせずに確かめる道具です。

- **読むもの**: ファイルの頭と、数字の表ごとに数 KB の抜き取りだけ。
- **分かること**: 足された・消えた層、広げられた形、書き換えられた値、余分なコードや pickle、会話の書式の変更。
- **分からないこと**: 安全の証明はできません。小さな値の書き換えは抜き取りで見逃すことがあります。
  この道具は構造の違いを見せるだけで、モデルが安全であることは証明できません。/ This tool shows structural differences. It cannot prove that a model is safe.

2 つの Hugging Face モデル、または手元のモデルフォルダの構造を比較します。
safetensors のヘッダーと小さな設定ファイルを読み、設定、層の指紋、テンソルの形・型、ファイル一覧の違いを表示します。
値の比較では、対応する名前と形が一致するテンソルから先頭 2,048 要素と、8,192 要素を超える場合は中央 2,048 要素を抜き取ります。
F32・F16・BF16 に対応し、異なる型は float32 に変換して丸めの範囲で比較します。同じ型はバイトの完全一致で比較します。
重みファイル全体はダウンロードせず、モデルのロードやリポジトリ内のコードの実行もしません。torch・GPU は不要です。

## インストールと使い方

Python 3.12 を推奨します（本体は Python 3.11 以上）。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/kizuna-intelligence/weightdiff
weightdiff Qwen/Qwen3.5-4B ccore/Qwen3.5-4B-3D-Coder --md out/coder.md --json out/coder.json
weightdiff ./model-a ./model-b --no-sample
```

[Space](https://huggingface.co/spaces/kizuna-intelligence/weightdiff) ではブラウザーから比較できます。

`--rev-a` / `--rev-b` でリビジョンを指定できます。Hub のブランチは比較開始時にコミットに固定します。
`--max-tensors N` は抜き取り数の上限（既定 160）です。超える場合は名前順で均等に間引きます。
`--no-sample` で値の比較を省略します。非公開モデルの認証は Hugging Face の既存の認証設定を使います。
設定 JSON と会話テンプレートは 16 MiB、ヘッダーは 100 MiB が上限です。
HTTP Range の応答が 206 でない場合は本体を読みません。失敗は取得事情または skipped として表示します。
通信切れ、読み取りタイムアウト、HTTP 429・5xx は、2・4・8 秒待って最大 3 回再試行します。

## 3 つの例の結果

A はすべて `Qwen/Qwen3.5-4B`。以下は既存の `out/*_v3.md` の実測値です（実行ごとに時間は変わります）。
「テンソル差」は名前・形・型の差があるテンソル数で、抜き取った値の差とは別です。

| B | 言語層 A → B | 幅 A → B | テンソル差 | 抜き取り same / different | 左上区画で一致 / 対象 | 秒 | 読んだバイト数 |
| --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `Qwen/Qwen3.5-4B` | 32 → 32 | 2560 → 2560 | 0 | 160 / 0 | — | 43.51 | 1,911,710 |
| `ccore/Qwen3.5-4B-3D-Coder` | 32 → 32 | 2560 → 2560 | 727 | 85 / 75 | — | 42.94 | 2,019,389 |
| `divagr1925/Qwen3.5-4B-Qwenputer` | 32 → 43 | 2560 → 3968 | 303 | 49 / 0 | 91 / 111 | 66.43 | 4,675,457 |

各比較の抜き取りは 160 個です。3D Coder と Qwenputer は別の並び `mtp.layers.` が 1 → 0。
Qwenputer の追加 11 層の位置は名前からは分かりません。左上区画で一致しなかった 20 個は、ほかの置き方を調べていません。
読み取り量はヘッダー・抜き取り・小ファイルの合計であり、キャッシュから読んだ小ファイルを含みます。

![Qwenputer の比較結果](docs/screenshot.png)

## 結果の読み方

- 同じ形の層が増えた場合、どこに増えたかは形から分かりません。位置は表示しません。
- 層数の異なる並びは指紋で比較し、その並びのテンソルを名前比較から除外します。
- same は同じ型なら抜き取ったバイトの一致、異なる型なら「抜き取った値は同じ（型の変換の丸めの範囲）」。全体の一致を意味しません。
- 接頭辞が違う並びも、指紋の重なりが小さい方の層数の 50% 以上なら組にします。層数が違う場合も、SequenceMatcher の equal 区間の層を抜き取ります。
- 形の指紋で組にできない並びは、テンソル名だけの指紋で同じ手順を試し、「名前で組（形が違う）」と表示します。幅が違う場合も対応した層を比べます。名前の指紋が既存と同じ追加層の位置は表示しません。
- 対応するテンソルの軸の大きさの変化を、テンソル数の多い順に最大 3 つ表示します。片側だけの `text_config.k` と反対側だけの `k` は同じ鍵として比較します。
- B の全軸が A 以上で少なくとも 1 軸が広いテンソルも、上限 160 個の対象です。2 次元は A の先頭・中央・最後の行全体と B の同じ行の先頭区画を読みます。1 次元は A 全体（8,192 要素を超えたら先頭と中央 2,048 要素）、3 次元以上は skipped です。
- `embedded_top_left` は抜き取った左上区画の一致です。`not_top_left` は「左上の区画には見つからない（ほかの置き方は調べていない）」を意味します。行全体の取得は通常の 2,048 要素の窓より長くなる場合があります。
- 層の外の名前には接頭辞の対応を当て、「名前の場所が変わった」テンソル数を表示します。並列取得は 16、型の比較対象外は skipped です。
- different は「値が違う（書き換えられている）」。用途や意図の判定はしません。
- `.bin` などの形式、Python ファイル・`auto_map`、会話書式・語彙の変化を気をつける点として表示します。
- 読んだバイト数はヘッダー・抜き取り・小ファイルの実測合計です。キャッシュから読んだ小ファイルを含み、HTTP のヘッダーと Hub API 一覧の通信量は含みません。

例: `Qwen/Qwen3.5-4B` 同士、`Qwen/Qwen3.5-4B` と `ccore/Qwen3.5-4B-3D-Coder`、
`Qwen/Qwen3.5-4B` と `divagr1925/Qwen3.5-4B-Qwenputer`。

## 構成と Space

`domain/` は純粋な比較だけ、`external/` はデータ取得だけを担当し、相互に import しません。
`application/` が収集・CLI・レポート・Gradio をつなぎます。試験は NumPy と safetensors で小さなフォルダを生成し、ネットを使いません。
`scripts/build_space.sh` はコード、依存一覧、ライセンスを `space/` にコピーします。
そのフォルダだけで Gradio が動き、パッケージ自体の pip install は不要です。無料 CPU 用の設定を同梱しています。

ソースから開発・試験する場合:

```bash
pip install -e '.[test]'
pytest --basetemp=out/pytest-tmp
bash scripts/build_space.sh
python space/app.py
```

生成した `space/` だけを使う場合は、新しい環境に `space/requirements.txt` を入れて `space/app.py` を起動します。

## ライセンス

[Apache-2.0](LICENSE)。

## English

Check a model against the base model it claims to come from — without downloading the weights.

- **What it reads**: Only file headers and a few KB of samples per tensor.
- **What it shows**: Added or removed layers, widened shapes, changed values, extra code or pickle files, chat-template changes.
- **What it can't do**: It cannot prove a model is safe. Small value changes can slip past sampling.
  This tool shows structural differences. It cannot prove that a model is safe.

Compare two Hugging Face models or local model folders using safetensors headers,
small configuration files, and bounded byte samples. No model is loaded, no repository
code executes, and no full weight file is downloaded. No torch or GPU is required.

After publication, install with:

```bash
pip install git+https://github.com/kizuna-intelligence/weightdiff
```

Browser UI: [Hugging Face Space](https://huggingface.co/spaces/kizuna-intelligence/weightdiff)
(available after publication). The table above records the three example comparisons
from the v3 reports: language-layer counts, hidden width, metadata differences,
sampled same/different counts, top-left matches, elapsed seconds and bytes read.
The screenshot shows the Qwenputer comparison.

Use `weightdiff A B --md out/report.md --json out/report.json`. Optional revisions:
`--rev-a R --rev-b R`. Disable sampling with `--no-sample`, or set its limit with
`--max-tensors N` (default 160, evenly subsampled in name order).

Reports show configuration differences, independent layer stacks and their shape
fingerprints, changed tensor metadata, sampled values, file differences, and points
to note. Added layers with an existing fingerprint have unknown positions. Tensors
in stacks whose layer counts differ are excluded from name comparison. Identical
samples do not imply identical weights. Failed reads and unsupported dtypes are explicitly skipped.
Connection failures, incomplete responses, read timeouts and HTTP 429/5xx retry up to
three times, waiting 2, 4 and 8 seconds.
Stacks with different prefixes are paired by fingerprint multiset overlap (at least 50%
of the smaller stack); equal layer intervals remain eligible for sampling even when
layer counts differ. Prefix mappings also match tensors outside layers.
Unmatched stacks fall back to sorted tensor-name fingerprints, with the same overlap
threshold and greedy pairing. Reports label these as paired by names (shapes differ).
The top three axis-size changes count each tensor once per size transition.
Exclusive `text_config.k` and `k` configuration keys are compared as aliases.
Samples use the first 2,048 elements and middle 2,048 for tensors over 8,192 elements.
F32/F16/BF16 pairs use float32 tolerance `2^-7 * max(abs(a), abs(b)) + 2^-14`;
identical dtypes use exact bytes. Requests run with 16 workers.
Wider B tensors share the 160-tensor limit: 2D comparisons read full A-width rows at
row 0, the middle A row and the last A row, at identical row indices in B. 1D reads
all A elements up to 8,192, otherwise first and middle 2,048; 3D and higher are skipped.
`embedded_top_left` indicates matching sampled blocks; `not_top_left` means not found
in the top-left block (other placements have not been checked). Full-row ranges can
exceed the usual element-window byte length. Neither result proves full-tensor equality.

Read-byte totals include inspected headers, samples and small files, including cache
reads; HTTP headers and Hub API listings are excluded. Run `pytest --basetemp=out/pytest-tmp`
for offline tests. Run `bash scripts/build_space.sh` then `python space/app.py` for the UI.
The generated `space/` is self-contained for a Gradio CPU Space. License: Apache-2.0.

---

## English

Check a model against the base model it claims to come from — without downloading the weights.

- **What it reads:** only safetensors headers, small config files, and a few KB of samples per tensor (HTTP Range requests).
- **What it shows:** added or removed layers, widened shapes, changed values, extra code or pickle files, chat-template changes.
- **What it can't do:** it cannot prove a model is safe. Small value changes can slip past sampling.

```bash
pip install git+https://github.com/kizuna-intelligence/weightdiff
weightdiff Qwen/Qwen3.5-4B ccore/Qwen3.5-4B-3D-Coder --md coder.md --json coder.json
weightdiff ./model-a ./model-b --no-sample
```

Try it in the browser: [Hugging Face Space](https://huggingface.co/spaces/kizuna-intelligence/weightdiff).

Example results (A = `Qwen/Qwen3.5-4B`; see the table above):

- `Qwen/Qwen3.5-4B` vs itself: no differences, 160/160 sampled tensors identical.
- `ccore/Qwen3.5-4B-3D-Coder`: same structure (32 layers, width 2560); 75 of 160 sampled tensors have different values (retrained).
- `divagr1925/Qwen3.5-4B-Qwenputer`: 32 → 43 layers, width 2560 → 3968; in 91 of 111 sampled widened tensors the original values sit in the top-left block, 0 changed.

How values are compared: tensors with the same name and shape are sampled (first 2,048 elements, plus the middle 2,048 when larger than 8,192). Same dtype → exact bytes; F32/F16/BF16 across dtypes → float32 within rounding. Widened tensors are checked for the original values in their top-left block. When added layers have the same shape as existing ones, their position is not shown. Nothing is loaded or executed; no torch or GPU needed.

License: Apache-2.0.
