# Verification on September 25, 2026

## September 26 calibration checks

The bounded calibration is specified in [CALIBRATION.md](CALIBRATION.md).
Before its launch, the existing 11-test suite passed locally with its GPU
test skipped, then passed on ml2 with all eight architecture/optimizer CUDA
resume cases enabled. The new `test_calibration.py` check passed locally and
its recipe/selection/resume checks also passed on ml2. Local controller
checks additionally exercise all seven stages, transfer of the selected
AdamW settings to every fallback, immutable restart, deadline enforcement,
paired-seed selection, checkpoint retention and proposed-final-plan gates.
The isolated tmux launcher test passes with calibration mode and scratch
cache initialization.

A full-size Mamba-2 300M run exercised the new AdaGrad-grafted Shampoo recipe
with momentum 0.9, refresh 50, fallback beta2 0.99 and fallback decay 0.01.
It trained one update, saved, resumed, and completed the second update at
sequence 2048, microbatch 1 and accumulation 16. The final checkpoint records
65,536 tokens. This checks allocation and resume at the actual training
batch; it does not establish convergence or optimal hyperparameters.

The server logs are `results/calibration-self-check.log`,
`results/calibration-gpu-check.log` and
`results/calibration-full-size-check.log`. The full-size checkpoints and
metrics are in `results/calibration-full-size-shampoo/`. The optimizer
library pin is unchanged; calibration only exposes its existing settings.

September 26 correction: the first pilot launch completed all eight Transformer
runs, but all eight Mamba runs failed during import. The persistent tmux server
had started before the scratch cache variables were exported, so Mamba's
TileLang dependency tried to create its cache under the missing home directory.
The earlier GPU checks explicitly sourced the environment inside their jobs;
they did not exercise this launcher failure. The launcher now sources the
environment inside its detached pane, and the regression test starts tmux
without the cache variable before invoking the launcher. `--retry-failed`
retries those imports in place and retains the completed Transformer runs.

The ml2 installation uses Python 3.12.12, PyTorch 2.10.0+cu128, Mamba
2.3.2.post1 and the pinned optimizer commit in `requirements.txt`. GPU checks
ran on one RTX PRO 6000 Blackwell Server Edition, 96 GB, with driver 595.84.

Passed checks:

- Interrupted data preparation reproduces the uninterrupted token files;
  altered token files fail checksum validation.
- All four optimizers reproduce model and optimizer state exactly across CPU
  checkpoint/resume. CUDA checks pass for both Transformer and Mamba-2, using
  relative tolerance 1e-4 and absolute tolerance 1e-5 for floating-point state.
- Validation preserves RNG state; checkpoint recovery restores the sampler,
  token schedule and separate Fisher RNG, and removes uncommitted metric rows.
- A committed final checkpoint recovers without repeating held-out evaluation.
- Queues resume paused trials and skip completed trials. Learning-rate
  selection uses terminal validation loss across every assigned seed.
- The launcher passes shell-quoting, exit-status and duplicate-session checks.
  An isolated real tmux test confirms Ctrl+C preserves the final log message
  and exit status. It does not touch the user's tmux sessions.
- CSV/PNG/PDF report generation passes; plots keep hardware groups separate.
  A repeated-seed example was rendered and inspected.

Two full-size checks used the prepared FineWeb-Edu artifact and the actual
training entry point:

| Configuration | Parameters | Work completed |
|---|---:|---|
| Mamba-2 300M, K-FAC+AdamW | 302,041,152 | Train, checkpoint, resume, finish |
| Transformer 300M, Shampoo+AdamW | 305,041,920 | Train, checkpoint, resume, finish |

Each check ran two updates at sequence 2048, microbatch 1, accumulation 1,
BF16 and refresh interval 1, with validation after each update. This exercises
full-size matrix decompositions, finite losses and complete-state reload.
It is not a convergence result or a controlled speed comparison. The small
tests cover all eight architecture/optimizer combinations; the two full-size
checks are not a claim that every full-size LR/batch configuration is stable.

Server artifacts under `/scratch/kravi-optimizer.DsO74p/optimizer_research`:

- `data/fineweb-edu-512m/`: 536,870,913 train tokens and 1,048,577 per holdout.
- `results/plans/pilot.json`: 16 real training pilots, 33,554,432 tokens each.
- `results/full-size-verification/`: full-size check metrics and checkpoints.
- `results/verification-report/`: plots and CSV exports from those checks.
- `results/readiness.json`: data/plan identifiers and verification revision.

Dataset ID:
`9b1235385999221d1485110ca80c1347958b8e695a37f32e65459641f4f78669`

Prepared pilot plan ID:
`98f3eb84ec0aa1c583734b2aa392dc34d0e02119bd28821571bf4accfbce8906`

The pilot queue was prepared but not launched during this setup. GPUs were
released after verification. Tomorrow's launch must recheck availability;
the prior idle snapshot is not a reservation. Launch and reconnect commands
are in [README.md](README.md).
