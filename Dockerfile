FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

# PyTorch supplies CUDA libraries; Triton needs a C compiler at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=4
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check \
    && python -m pip freeze > /app/environment.txt
COPY benchmark_models.py hardware_benchmark.py ./
RUN python -c "from mamba_ssm.modules.mamba2 import Mamba2; import causal_conv1d" \
    && python hardware_benchmark.py --self-test
# Starting a container without arguments only runs the CPU self-check.
CMD ["python", "hardware_benchmark.py", "--self-test"]
