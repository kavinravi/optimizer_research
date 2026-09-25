# Compare an A100 and RTX PRO Blackwell within $10

Run one pod at a time. Start with one **A100 SXM 80GB**, download its results,
terminate it, then repeat on one **RTX PRO 6000 Blackwell 96GB**. Choose whole
GPUs on on-demand Pods, not MIG partitions, Serverless, or interruptible Spot.

As checked September 20, 2026, RunPod lists these at $1.59/hour and $2.09/hour.
An hour on each totals $3.68 in GPU charges. Two hours on each totals $7.36,
before storage. Check the displayed rates before launching. Set a **75-minute
alarm starting at deployment** for each pod and terminate at that point even
if tests are unfinished. At the listed rates that is $4.60 in GPU charges
across both pods, leaving room for storage and troubleshooting. A script
timeout stops computation but DOES NOT stop RunPod billing.

These are cloud hardware proxies. The exact Blackwell model/edition in `ml2`
and the A100 memory/power configuration in `dgx0` still need confirmation.
This measures one GPU at a time, not eight-GPU DGX scaling or campus contention.

## 1. Prepare the files before renting

The archive `runpod-hardware-benchmark.tar.gz` contains this guide,
`hardware_benchmark.py`, `benchmark_models.py`, and `setup_runpod.sh`.
No dataset or checkpoint download is needed. The benchmark uses synthetic
token sequences to measure compute throughput, not loss convergence.

## 2. Deploy the A100 pod

In RunPod, select **Pods → Deploy**, one **A100 SXM 80GB**, on-demand.
Use the official RunPod **PyTorch 2.8 / Python 3.11 / CUDA 12.8** template with
Jupyter enabled. Use the identical template for Blackwell. The setup script
creates a separate PyTorch 2.10.0 CUDA 12.8 environment with matching prebuilt
Mamba and causal-convolution wheels. It does not compile their CUDA extensions.

Allow roughly 40 GB of workspace for the Python environment and downloads.
A network volume is unnecessary for this short test. Leave auto-pay disabled
if you intend to limit spending to the existing balance.

Open **Connect → JupyterLab**, navigate to `/workspace`, and upload the archive
using the file browser. Open a Jupyter terminal and run:

```bash
cd /workspace
tar -xzf runpod-hardware-benchmark.tar.gz
cd runpod-hardware-benchmark
set -o pipefail
timeout --kill-after=15s 15m bash setup_runpod.sh 2>&1 | tee setup.log
```

Stop if setup fails or times out. Download `setup.log` and terminate the pod
while resolving the problem. Do not let an idle pod burn credits while debugging.
The setup includes eight small GPU checks, covering both architectures and all
four optimizers. All eight must pass before the full benchmark.

## 3. Run the benchmark

```bash
timeout --kill-after=15s 50m .venv/bin/python -u hardware_benchmark.py \
  --output results 2>&1 | tee benchmark.log
```

The defaults cover these provisional study shapes, copied from the SLM training
notebooks with reduced width/depth:

| Architecture | Approximate size | Actual parameters | Width | Layers |
|---|---|---:|---:|---:|
| Transformer | 150M | 153,271,808 | 768 | 16 |
| Mamba-2 | 150M | 151,360,368 | 768 | 30 |
| Transformer | 300M | 305,041,920 | 1024 | 20 |
| Mamba-2 | 300M | 302,041,152 | 1024 | 38 |

Each model runs AdamW, Muon, Shampoo, and sampled-Fisher K-FAC. SOAP is available
explicitly via `--optimizers ... soap`, but is not included by default. There
are no 538M runs. The four optimizers use identical matrix selections and
AdamW fallback parameters within each architecture.

Configuration is fixed across GPUs:

- Sequence length 2048, microbatch 1, accumulation 4: 8192 training tokens per update.
- BF16 autocast, FP32 parameters, FP64 Shampoo/K-FAC decompositions, TF32 disabled.
- Activation checkpointing enabled, `torch.compile` disabled, reference optimizer backend.
- Mamba `use_mem_eff_path=False` for every optimizer so K-FAC hooks can collect factors.
- Matrix refresh every 10 steps; sampled K-FAC statistics collected on those steps.
- Ten warm-up steps followed by 30 measured steps, including three refresh cycles.
- Each case has a 180-second process timeout. Failures/timeouts are recorded, never treated as fast results.

The CPU timer synchronizes CUDA at step boundaries. Mean step time includes
forward/backward, sampled-Fisher work, clipping, and optimizer updates. CUDA
events separately record optimizer and sampled-Fisher time. Compilation during
warm-up is excluded from the reported throughput but consumes rental time.

Logs print each completed case's tokens/second and peak allocated memory. Each
case writes a JSON result immediately, so successful cases survive later failures.

## 4. Save results and terminate before switching GPUs

```bash
tar -czf /workspace/a100-results.tar.gz results smoke environment.txt setup.log benchmark.log
```

Download `a100-results.tar.gz` through Jupyter's file browser. Then use the RunPod
console to **terminate the pod**. Merely exiting the terminal or finishing the
script does not release the GPU. Stopped pods can still incur storage charges.
Termination deletes local pod data, so download first.

Repeat steps 2–4 on **RTX PRO 6000 Blackwell 96GB** using the same archive,
template and commands. Name the second results archive `blackwell-results.tar.gz`.
Avoid RTX 6000 Ada or the 24/48 GB MIG variants; those are different devices.

## 5. Interpret both results

For each matching architecture, size and optimizer, compare `mean_step_seconds`
and `tokens_per_second`. Compute `A100 seconds / Blackwell seconds`: above 1
means Blackwell was faster for that case; below 1 means A100 was faster.
Keep the optimizer results separate. Averaging everything could hide an A100
advantage for FP64-heavy optimizers.

Check `cycle_mean_seconds` for variation. A small difference with substantial
variation is inconclusive; rerun the relevant case only if credit remains.
Verify matching `config`, package versions, benchmark/model hashes, and optimizer
source hashes. Hardware metadata records the actual GPU, memory, driver and
power limit. A timeout or out-of-memory error is not a throughput measurement.

This is a throughput screen with random initialization and synthetic data.
It excludes FineWeb-Edu loading, validation, checkpoint I/O and convergence.
The chosen accumulation and refresh cadence are provisional; changing them
changes the fraction of time spent in FP64 work and can change the hardware ranking.
The controlled eager/unfused execution path also does not measure each GPU's
best achievable performance with independently optimized model implementations.
Use the results to choose a platform for pilots; establish training quality and
final study settings with FineWeb-Edu afterwards.

Local verification: the CPU self-check and all eight small GPU cases passed on
an RTX 5090. A four-step, 151M-parameter Mamba/K-FAC check also passed at sequence
length 256. Those checks used the existing local PyTorch 2.13 environment; the
pinned RunPod installation and the complete default grid have not yet been run.
They do not establish an A100 versus RTX PRO performance ranking.

Sources checked September 20, 2026:

- [RunPod pricing](https://www.runpod.io/pricing)
- [RunPod PyTorch template](https://www.runpod.io/articles/guides/pytorch-2-8-cuda-12-8)
- [Pod lifecycle and storage](https://docs.runpod.io/pods/manage-pods)
- [PyTorch 2.10 CUDA 12.8 installation](https://pytorch.org/get-started/previous-versions/)
- [Mamba prebuilt wheels](https://github.com/state-spaces/mamba/releases/expanded_assets/v2.3.2.post1)
- [Causal-convolution prebuilt wheels](https://github.com/Dao-AILab/causal-conv1d/releases/expanded_assets/v1.7.0)
