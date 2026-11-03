#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python - "$root" <<'PY'
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = root / 'space' / 'src'
if target.exists():
    shutil.rmtree(target)
shutil.copytree(root / 'src', target,
                ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
shutil.copyfile(root / 'requirements.txt', root / 'space' / 'requirements.txt')
shutil.copyfile(root / 'LICENSE', root / 'space' / 'LICENSE')
PY
