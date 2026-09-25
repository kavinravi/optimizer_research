#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

# Use the RunPod Python 3.11 PyTorch template on both machines.
python -c 'import platform, sys; assert sys.version_info[:2] == (3, 11), "Choose the Python 3.11 template"; assert platform.machine() == "x86_64"'
python -m venv .venv
.venv/bin/python -m pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install --no-cache-dir packaging==25.0 einops==0.8.1 transformers==4.51.3 ninja==1.11.1.4
# Direct upstream wheels fail promptly if unavailable, rather than compiling for an hour.
printf 'torch==2.10.0\ntransformers==4.51.3\n' > constraints.txt
.venv/bin/python -m pip install --no-cache-dir --constraint constraints.txt \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl' \
  'https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl' \
  'git+https://github.com/kavinravi/pytorch-opt.git@b58ab855d6154b050e5bcc115be62a45ff24a469'
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > environment.txt
.venv/bin/python -c 'import torch; from mamba_ssm.modules.mamba2 import Mamba2; from pytorch_opt import matrix_param_groups; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda)'
.venv/bin/python hardware_benchmark.py --tiny --sequence 32 --accumulation 1 --refresh 2 --cycles 1 --sizes 150m --output smoke
.venv/bin/python - <<'PY'
import json
from pathlib import Path
results = list(Path('smoke').glob('*.json'))
assert len(results) == 8
failed = [str(p) for p in results if json.loads(p.read_text())['status'] != 'ok']
assert not failed, f'Smoke checks failed: {failed}. Stop here and inspect the setup log.'
PY
