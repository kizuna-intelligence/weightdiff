---
title: weightdiff
emoji: 🔍
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
license: apache-2.0
short_description: Compare a model with its base model without downloading the weights
suggested_hardware: cpu-basic
pinned: false
---

# weightdiff

この道具は構造の違いを見せるだけで、モデルが安全であることは証明できません。/ This tool shows structural differences. It cannot prove that a model is safe.

Compare two Hugging Face model IDs using safetensors headers, small configuration files,
and at most two windows of 2,048 sampled elements per selected tensor (up to 16 KB per model).
The default limit is 160 tensors. F32, F16 and BF16 are supported; different dtypes are
compared within conversion-rounding tolerance. Identical samples do not establish
identical weights. No model code is executed. CPU only.

Examples: `Qwen/Qwen3.5-4B`, `ccore/Qwen3.5-4B-3D-Coder`, `divagr1925/Qwen3.5-4B-Qwenputer`.
