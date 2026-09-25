# Optimizer research

Train Transformer and Mamba-2 language models from scratch on FineWeb-Edu at
approximately 150M and 300M parameters. Compare AdamW, Muon, Shampoo, and
sampled-Fisher K-FAC. SOAP is an explicit replacement if K-FAC proves
infeasible, not a fifth primary arm. Optimizers are pinned to
[pytorch-opt](https://github.com/kavinravi/pytorch-opt/tree/b58ab855d6154b050e5bcc115be62a45ff24a469).

`train.py` performs actual next-token training, validation and checkpointing.
`study.py` creates immutable trial plans, runs one independent trial per GPU,
and selects learning rates from validation loss. `report.py` exports curves,
CSV summaries and PDF/PNG plots. The older synthetic throughput script is
optional and is not part of the training launch.

## Start on ml2 from any laptop

Connect to the Chapman VPN if campus Wi-Fi cannot resolve the server. Run:

```bash
ssh kravi@ml2.chapman.edu
```

Then, on **ml2**, use the existing installation:

```bash
source /scratch/kravi-optimizer.DsO74p/env.sh
cd /scratch/kravi-optimizer.DsO74p/optimizer_research
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv
```

The native Python environment works without Docker or a home directory.
Nothing needs to run on the laptop except SSH. The study files and processes
remain on ml2 when the desktop or laptop disconnects.

For the prepared pilot plan, choose **two currently free** GPUs. These are
ml2's GPU 0 and GPU 1 UUIDs; verify they are still free before using them:

```bash
bash launch_study.sh results/plans/pilot.json \
  GPU-09f93eaf-a6fa-9651-59fb-b7ddc3656f3a \
  GPU-45699d16-5721-64f5-3624-c657bbbec7e4
```

This starts a real FineWeb-Edu pilot queue in detached tmux. The default plan
has one 33,554,432-token pilot per architecture, size and optimizer. These
runs establish stability, memory use, attainable losses and training cost.
They are not the tuned, repeated final comparison. The queue uses at most
two GPUs and checks for existing compute processes before every trial.
GPU availability checks are advisory on this first-come, first-served server;
they cannot reserve a device against another user starting a job.

The launcher defaults to an **eight-hour queue window**, then finishes each
active update and checkpoints. Repeat the same command after the session
ends to resume unfinished trials and skip completed ones. To choose another
window, prefix the launch command with `STUDY_HOURS=4`. A failed trial stays
failed until inspected; it is never silently counted as a completed result.

To see the running queue, from either computer:

```bash
tmux attach -t optimizer-training
```

Detach with **Ctrl+A, then d**. Herdr uses Ctrl+B, so this tmux session uses
Ctrl+A. For scrollback use Ctrl+A, then `[`, and Page Up; Escape exits the
default copy mode. Ctrl+C inside the training session asks the queue to stop
and save after the current optimizer update. Matrix inversions can make that
update slow; wait for the checkpoint message. The session disappears when
the queue exits. Inspect its logs without attaching:

```bash
tail -n 40 results/queue.log
cat results/queue.exit
.venv/bin/python report.py --results results/training --output results/report
```

`queue.exit` exists after a launch ends. A nonzero code means inspect the
reported trial's `stderr.log`. Each trial also has `console.log`,
`metrics.jsonl`, `status.json`, and its committed checkpoint pointer
`latest.json`. Compiler warnings stay in `stderr.log`.

The existing `optimizer-study` tmux session is a separate control shell.
It can stay open while `optimizer-training` runs. A second SSH connection
can inspect, attach or stop the same server-side sessions. To stop the queue
from that shell without attaching:

```bash
tmux send-keys -t optimizer-training:0.0 C-c
```

Do not pull code or change dependencies while study runs are active. Training
source hashes are frozen in each plan and checkpoint. Code changes require
new plans; exact resume deliberately rejects a different implementation.

## Data preparation and planning

The repository includes the frozen summer 50,000-token BPE tokenizer, with
its [provenance and checksum](tokenizer/README.md). New training uses only
FineWeb-Edu. It does not reuse the summer's mixed corpus or validation set.

```bash
.venv/bin/python prepare_data.py --output data/fineweb-edu-512m
.venv/bin/python prepare_data.py --output data/fineweb-edu-512m --verify
mkdir -p results/plans
.venv/bin/python study.py plan --phase pilot --output results/plans/pilot.json
```

Preparation downloads pinned `sample-10BT` parquet shards from
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
It assigns documents to train/validation/test using a stable text hash with
98/1/1 proportions. NFC and whitespace-equivalent copies stay in the same
split. This prevents exact normalized copies crossing the split; it is not
a claim of near-duplicate or benchmark decontamination. Each document ends
with EOS; batches pack documents without resetting attention or Mamba state
at EOS. All optimizers receive the same packed-block order for a given seed.

The default artifact has 536,870,913 training tokens and 1,048,577 tokens in
each holdout. The extra token supplies the final next-token target.
Preparation checkpoints its shard/row position and output lengths, so rerun
the identical command after interruption. Completed artifacts are immutable
and checksummed. No Hugging Face login is needed for this public dataset.

`study.json` holds provisional pilot budgets, candidate rates and seeds.
Change it before creating a plan. `--sizes 150m` creates a smaller initial
pilot; `--architectures`, `--tokens` and `--seeds` also limit a new plan.
Existing plans are never overwritten. Actual run count follows the chosen
scope and tuning budget; neither 40 nor 58 is a requirement. There are no
538M runs in this study.

After pilots, decide the main token budget and attainable loss targets.
**Prepare enough data for that budget before LR tuning**, then use that same
artifact for tuning and final runs. For example, if the eventual budget is
one billion tokens, choose a step-aligned total and prepare at least that
many tokens plus one in a new directory. Set `data` in `study.json` to that
artifact. This intentionally prevents transferring selected rates across
silently changed training/holdout data. A new artifact can use the same
pinned source, split rule and tokenizer, but still needs its own tuning.
By default training rejects a budget larger than one prepared epoch.

## Learning-rate tuning and final runs

Tune the AdamW control first. Its selected rate fixes the common AdamW
fallback rate for the remaining optimizers within each architecture and size.
Then tune the matrix optimizer rates. Every arm receives the same number of
LR candidates, token budget and paired seeds. This is a conditional search
with a shared fallback rate, not a separate two-dimensional search per arm.
The illustrative grids in `study.json` are starting ranges, not proven optima.

```bash
.venv/bin/python study.py plan --phase tune-adamw --output results/plans/tune-adamw.json
# Launch this plan with launch_study.sh and wait for its trials to finish.
.venv/bin/python study.py select --plans results/plans/tune-adamw.json \
  --output results/plans/adamw-selection.json
.venv/bin/python study.py plan --phase tune-others \
  --selection results/plans/adamw-selection.json --output results/plans/tune-others.json
# Launch tune-others.json and wait for its trials to finish.
.venv/bin/python study.py select --plans results/plans/tune-adamw.json results/plans/tune-others.json \
  --output results/plans/final-selection.json
```

Selection uses mean **terminal validation loss**, never best intermediate
loss or test loss. Every candidate must finish every assigned seed; a failed
seed makes that candidate ineligible. Pending or paused runs block selection.
A grid-boundary winner is flagged for review. If ranges need expansion, give
affected arms equal additional search budgets and record failures and cost.
The queue records attempt durations, including failed attempts.

Final comparisons require at least two seeds distinct from tuning seeds;
the default is three. Choose the final token budget explicitly after pilots.
The example variable below must be set to that budget, divisible by 32,768
under the default batch configuration:

```bash
.venv/bin/python study.py plan --phase final --tokens "$FINAL_TOKENS" \
  --selection results/plans/final-selection.json --output results/plans/final.json
```

Run that plan with the same launcher. The held-out test split is evaluated at
the end of frozen final runs only. Do not use it for further tuning. Loss
2.5 is provisional; reports have no default success threshold. To inspect a
pilot-established target, pass `report.py --target VALUE`. Target crossings
are the first observed evaluation at or below the target, not interpolated
training steps. A missing crossing stays missing.

## Models, routing and measurements

| Model | Parameters at sequence 2048 | Width | Layers |
|---|---:|---:|---:|
| Transformer 150M | 153,271,808 | 768 | 16 |
| Mamba-2 150M | 151,360,368 | 768 | 30 |
| Transformer 300M | 305,041,920 | 1024 | 20 |
| Mamba-2 300M | 302,041,152 | 1024 | 38 |

Defaults are BF16 autocast with FP32 weights, sequence 2048, microbatch 1,
16 accumulated microbatches, activation checkpointing and gradient clipping
at 1.0. The schedule is token-based warmup followed by cosine decay. The
reference optimizer backend, eager execution and Mamba's
`use_mem_eff_path=False` apply to every arm. TF32 is disabled. K-FAC uses a
separate sampled-label RNG and refreshes curvature/inverses every ten
updates; Shampoo refreshes matrix roots on the same configured cadence.
FP64 decompositions can be costly on Blackwell; pilots measure that cost.

All non-head linear weights use the selected matrix optimizer. Embeddings,
the tied output head, normalization, biases and Mamba's convolution/state
parameters use AdamW in **every arm**. The AdamW control uses AdamW everywhere
with the same named groups and decay rules. Every run records the exact
parameter names and counts for each group. Report these methods as
Muon+AdamW, Shampoo+AdamW and K-FAC+AdamW, with AdamW as the control.

Mamba's input projection contains heterogeneous state-space channels.
`selection: "mamba_output"` is an optional routing ablation, applied to
**all four arms** in a Mamba-only plan. It must have its own tuning and final
comparison. It is not silently substituted for the primary all-linear policy.

`metrics.jsonl` records validation cross-entropy against tokens and two clocks:

- `train_seconds` includes batch retrieval, transfer, forward/backward,
  sampled-Fisher work, optimizer updates and refresh steps, with GPU
  synchronization. It excludes validation and checkpoint I/O. First-use JIT
  compilation during an update remains included; no warmup steps disappear.
- `wall_seconds` measures cumulative elapsed time inside the trainer, including
  setup, evaluation, checkpointing and resume setup. It excludes time between
  stopped processes and imports before entering the trainer. Queue attempt
  durations include that process startup. Reports plot both clocks and tokens.

A crash can lose work since the last commit. Resume rolls metrics back to the
committed state and keeps discarded rows separately. Queue attempt logs retain
whole-process costs, including unsuccessful work. A forcibly killed queue may
lack its final attempt record, so those logs are a lower bound in that case.
Compare hardware costs using both committed curves and attempt records.
Other users' workloads and clock/thermal conditions can affect timings;
repeat trials and retain server context instead of treating shared-server
measurements as isolated hardware specifications.

Reports separate hardware/software stacks, model configurations, phases,
data fingerprints and source revisions. They show individual seed curves
and terminal means/sample standard deviations, with completion/failure counts.
Inspect seed sets and unfinished runs before comparing means. A one-seed
pilot has no repeated-seed uncertainty estimate. RTX 5090 summer timings,
ml2 Blackwell timings and dgx0 A100 timings remain separate.

## Checkpoint recovery and outputs

Each committed checkpoint contains weights, optimizer state including matrix
factors/inverses and fallback moments, token schedule position, packed-data
cursor, Python/NumPy/Torch/CUDA RNGs, the separate Fisher RNG, and accumulated
clocks. The trainer writes and fsyncs a new checkpoint before publishing
`latest.json`; the previous committed checkpoint is retained. Validation
preserves training RNG state. Exact resume requires identical data, config,
code and the same GPU type/software stack.

A paused run resumes automatically through the same queue command. For a
failed run, inspect its `stderr.log` first, then explicitly retry through:

```bash
.venv/bin/python study.py run --plan results/plans/pilot.json \
  --gpus GPU-REPLACE-WITH-FREE-UUID --hours 4 --retry-failed
```

Use tmux if starting this command manually. Retry resumes the last committed
checkpoint. Failure before the first checkpoint requires a fresh run directory;
move the failed directory aside to preserve diagnostics. For physical damage
to the latest checkpoint, stop the queue and preserve the damaged files before
replacing `latest.json` with `previous.json`, then retry. Checkpoints are trusted
local PyTorch artifacts; do not load downloaded third-party pickle files.

Data, logs, plans, plots and checkpoints are ignored by Git. Copy results from
your laptop, for example after generating a report:

```bash
scp -r kravi@ml2.chapman.edu:/scratch/kravi-optimizer.DsO74p/optimizer_research/results/report ./ml2-report
```

Scratch is working storage, not a promised backup. Back up completed results
and important checkpoints to storage approved by the university. Check `df -h
/scratch` during longer campaigns. Stop/checkpoint before the announced October
3–4 downtime. Close idle containers and remote editor sessions when finished.

## Installation and checks

The existing ml2 environment is ready under the scratch path above. A fresh
Linux x86-64 installation needs Python 3.12, a C compiler for Triton, and the
NVIDIA driver for GPU work:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
CUDA_VISIBLE_DEVICES='' .venv/bin/python test_training.py
CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-FREE-UUID .venv/bin/python test_training.py --gpu
```

Dependencies pin PyTorch 2.10/CUDA 12.8, Mamba 2.3.2.post1, causal-conv1d 1.7.0
and the optimizer commit. Hashed binary wheels avoid a source build. Each run
saves installed versions; reuse the same environment across comparisons.
An optional container runs the same tests during its build:

```bash
docker build --platform linux/amd64 -t optimizer-research:ml2 .
```

Docker permission is currently unavailable to this ml2 account and is not
needed. The native environment uses scratch caches without changing `HOME`.
The `_POSIX_C_SOURCE` redefinition printed by Triton's host compilation is a
warning; the exit code and trainer tests determine whether compilation worked.
The [verification record](VERIFICATION.md) lists the completed checks and the
prepared ml2 data/plan identifiers.

## Research context

[Muon Meets Mamba](https://arxiv.org/abs/2608.03941) already studies Muon+AdamW
for Mamba-2 and reports that projection selection matters, with output-only
routing performing strongly. This study must not claim the first Muon/Mamba
comparison. Its proposed contribution is a controlled architecture-by-scale
comparison including a natural-gradient arm, shared fallback routing, direct
FineWeb-Edu LR tuning, repeated final runs and validation loss against both
tokens and measured training time. The routing ablation tests whether that
prior observation transfers across optimizers and scales.

Historical compute proposals and the [RunPod guide](RUNPOD_BENCHMARK.md) are
context only. The actual allocation is shared ml2 access, normally two RTX
PRO 6000 Blackwell GPUs. Final budgets and targets follow the pilots and the
university's availability.
