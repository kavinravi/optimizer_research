# Optimizer research

Transformer and Mamba-2 optimizer study at approximately 150M and 300M
parameters, using FineWeb-Edu. The primary platform is two GPUs on Chapman's
ml2, with one independent trial per GPU. Start with one GPU for setup checks.

This repository owns the study setup and benchmark. Optimizers come from
[`kavinravi/pytorch-opt`](https://github.com/kavinravi/pytorch-opt), pinned to
commit `b58ab855d6154b050e5bcc115be62a45ff24a469`. Building this project does
not modify that library or the SLM inference repository.

## Current scope

The runnable entry point is a **synthetic hardware benchmark**, including
small compatibility checks. It does not download FineWeb-Edu, measure
validation loss, save resumable training checkpoints, or launch the main
study. A successful check establishes that the model and optimizer can run;
it does not establish convergence.

The main experiment still needs a FineWeb-Edu training entry point, fixed
tokenizer and train/validation splits, complete resume state, and a pilot
protocol before long runs. Establish attainable loss targets in those pilots;
2.5 is provisional. Compare validation loss against both tokens and measured
training time, tune learning rates fairly, and repeat final comparisons.

Compare AdamW, Muon, Shampoo, and K-FAC. K-FAC provides the natural-gradient
arm; SOAP replaces it only if K-FAC proves infeasible. Use the same explicitly
selected module weights and AdamW fallback across arms within each
architecture. Keep final timing comparisons on the same GPU type. There are
no planned 538M runs and no fixed run count. Historical RTX 5090 timings
remain separate, and the contribution must account for *Muon Meets Mamba*.

## Resume from a laptop

Connect to the Chapman VPN if campus Wi-Fi alone does not give access, then
run this in your own terminal. Enter the Chapman password at the SSH prompt.

```bash
ssh kravi@ml2.chapman.edu
```

A DNS failure or timeout happens before authentication. This repository does
not contain passwords or change the university's access configuration.

On ml2:

```bash
git clone https://github.com/kavinravi/optimizer_research.git
cd optimizer_research
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv
nvtop
docker version
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

Use `nvtop` and the process list to select a free GPU. The server is currently
first-come, first-served; the CPU-only dashboard does not show GPU contention.
The container needs a working NVIDIA driver and Docker's NVIDIA GPU support
on the host. If Docker is unavailable or access is denied, confirm the
supported container runtime with the administrator before proceeding.

## Build and check

Run from the repository root. Build on the Linux GPU machine; the image
targets Linux x86-64 even if your laptop is an Apple Silicon Mac.

```bash
set -o pipefail
docker build --platform linux/amd64 -t optimizer-research:ml2 . 2>&1 | tee build.log
```

The build installs the pinned dependencies, checks their consistency,
imports Mamba-2, and runs a CPU check of optimizer updates and error handling.
It does not reserve a GPU. The first build downloads several GB; allow time
for that before a meeting. Subsequent builds reuse Docker's cache.

Copy the UUID of one free GPU from `nvidia-smi`, replacing the example below:

```bash
bash run_ml2.sh GPU-REPLACE-WITH-FREE-GPU-UUID smoke
```

The smoke check runs eight tiny cases: Transformer and Mamba-2, each with
AdamW, Muon, Shampoo, and K-FAC. It has a ten-minute total limit and a
90-second limit per case. Every case must pass. A failure produces a nonzero
exit status and retains its diagnostic JSON and log.

After the smoke check passes, the optional hardware benchmark is:

```bash
bash run_ml2.sh GPU-REPLACE-WITH-FREE-GPU-UUID benchmark
```

This covers both model sizes and all four optimizers, with a 55-minute total
limit and 180 seconds per case. A timed-out case is a failure, not a speed
measurement. Inspect failures before changing limits or launching more work.
This command is still a synthetic throughput screen, not a FineWeb-Edu pilot.

## Results and cleanup

Every invocation creates a new directory under `results/`, containing logs,
per-case measurements, the study revision, image metadata, installed package
versions, and GPU details. These files stay on the host after the container
exits and are excluded from Git. Download them from your laptop, for example:

```bash
scp -r kravi@ml2.chapman.edu:~/optimizer_research/results ./ml2-results
```

The runner exposes one chosen GPU, limits CPU and host memory, runs as your
user, and uses `--rm`. It stops its own container on interruption. The
in-container timeout also bounds the job if the SSH client disappears.
For these short checks, stay connected until completion. To stop a run from
another SSH session, use the exact container name printed when it started:

```bash
docker stop THE-PRINTED-CONTAINER-NAME
```

Stop your idle containers and close your VS Code remote sessions when done.
Do not stop other users' containers or run a system-wide Docker prune.
Plan to finish or checkpoint long jobs before the announced October 3–4
downtime. The synthetic benchmark has no resume state.

## Benchmark settings

| Model | Parameters at sequence length 2048 | Width | Layers |
|---|---:|---:|---:|
| Transformer 150M | 153,271,808 | 768 | 16 |
| Mamba-2 150M | 151,360,368 | 768 | 30 |
| Transformer 300M | 305,041,920 | 1024 | 20 |
| Mamba-2 300M | 302,041,152 | 1024 | 38 |

Defaults: sequence length 2048, microbatch 1, accumulation 4, BF16 autocast
with FP32 parameters, activation checkpointing, eager execution, and the
reference optimizer backend. Shampoo and K-FAC matrix decompositions use
FP64. Mamba uses `use_mem_eff_path=False` in every arm to expose the module
hooks required by K-FAC. These choices affect measured speed.

Each case warms up for ten steps and measures thirty more, including three
matrix-refresh cycles. The average includes expensive refresh steps. The
fixed learning rate is for throughput screening only, not the main study's
learning-rate comparison. The benchmark excludes data loading, validation
and checkpoint I/O, and uses synthetic tokens with a provisional 50,000-token
vocabulary rather than loading the study tokenizer.

## Environment and verification

The image pins a Python 3.12 base digest, PyTorch 2.10 with CUDA 12.8, and
upstream binary wheels for Mamba 2.3.2.post1 and causal-conv1d 1.7.0. These
extensions' release builds include Blackwell support. No CUDA compilation
is needed during setup. The host still supplies the GPU driver.

Verified September 25, 2026: the image builds, package consistency and CPU
checks pass, and all eight tiny GPU cases pass inside the container on an
RTX 5090, including Mamba-2 with sampled-Fisher K-FAC. The runner's GPU
selection, cleanup, and failure exit status also pass their local check.
The full-size benchmark, ml2 login, and ml2's installed runtime have not yet
been verified. No FineWeb-Edu training has started.

The base image and main dependencies are pinned. Transitive package versions
are captured in `/app/environment.txt` and copied into each results directory.
Reuse the same built image for controlled comparisons; a fresh build can
resolve newer transitive dependencies and must be checked again.

Checks that do not need a GPU:

```bash
python test_runner.py
python hardware_benchmark.py --self-test
```

The older [RunPod guide](RUNPOD_BENCHMARK.md) and `setup_runpod.sh` describe
an optional cloud comparison, with a separate Python 3.11 environment.
They are not needed for ml2. RunPod pricing in that guide is historical.

References:

- [NVIDIA Container Toolkit: running a GPU container](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html)
- [Mamba release build configuration](https://github.com/state-spaces/mamba/blob/v2.3.2.post1/.github/workflows/publish.yaml)
- [Causal-convolution release build configuration](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/.github/workflows/publish.yaml)
