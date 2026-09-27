# Calibration before the main comparison

This protocol was fixed on September 26, 2026, after the 16 initial pilots.
Those pilots completed 33,554,432 tokens each. They established executable
training and checkpoint recovery for both architectures and all four
optimizers. Their learning rates were provisional, so their losses do not
establish a final optimizer ranking.

The calibration uses real FineWeb-Edu training on ml2. It covers Transformer
and Mamba-2 at approximately 150M and 300M parameters. There are no 538M runs,
and no RTX 5090 or A100 measurements enter its timing comparisons.

## What an "inner optimizer" means here

There are three separate choices:

1. The selected hidden matrices receive the named optimizer's update.
   Shampoo computes a two-sided matrix-preconditioned gradient. K-FAC
   computes a damped, Kronecker-factored Fisher-preconditioned gradient.
   Neither current implementation feeds that gradient through AdamW.
   Momentum, when enabled, accumulates the preconditioned direction.
2. Grafting rescales a Shampoo direction to another update's norm. It does
   not replace its direction with that optimizer's direction. We test
   ungrafted Shampoo and the installed implementation's AdaGrad graft.
3. Parameters outside the selected matrix set use the common AdamW fallback.
   These include tied embeddings/head weights, vectors, normalization,
   biases, and Mamba's depthwise convolution and state parameters.

We **do not know that AdamW is the best possible fallback**. We use it as the
same experimental control in every advanced arm, following the earlier study
decision. We first calibrate the full AdamW control, then freeze its learning
rate, beta2 and decay as the fallback settings for that architecture and size.
This does not establish that the full-model optimum is also the optimum for
the fallback subset.

Muon is already the tested update on the eligible hidden matrices. Its
[reference implementation](https://github.com/KellerJordan/Muon) assigns
embeddings, heads and scalar/vector parameters to AdamW. Substituting Muon
for our entire fallback would require new rules for non-matrix parameters
and tied weights. It is not an interchangeable setting of these algorithms.

The native K-FAC and ungrafted Shampoo recipes supply the requested comparison
without an adaptive optimizer after the matrix preconditioner. This is still
an approximation: Shampoo uses gradient outer products, and K-FAC uses a
factored Fisher approximation, not a full-model Hessian inverse. Extending
either method to every parameter would change the shared routing policy
and require additional curvature definitions and validation.

The original [K-FAC paper](https://arxiv.org/abs/1503.05671) defines its Fisher
approximation. [Distributed Shampoo](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md)
supports several grafting choices, including Adam. Our pinned implementation
supports none, SGD and AdaGrad; it does not implement Adam grafting. We do not
silently substitute an unverified implementation during calibration.
[SOAP](https://arxiv.org/abs/2409.11321) applies Adam in a Shampoo-derived
eigenbasis. That is a different algorithm, not a switch that makes native
Shampoo or K-FAC more "pure". SOAP remains a contingency if K-FAC becomes
infeasible; successful Mamba pilots have not required it.

## Search and selection

The executable specification is [calibration.json](calibration.json).
[calibration.py](calibration.py) writes immutable plans before each stage,
runs them with the existing queue, and records rankings from terminal
validation loss. It never starts final training.

For every architecture, size and optimizer:

| Stage | Candidates and seeds | Tokens per run | Purpose |
|---|---|---:|---|
| Screen | Three recipes, four LRs, seed 3619 | 16,777,216 | Eliminate poor or unstable candidates |
| Wider bracket | Same recipes, two additional LRs, seed 3619 | 16,777,216 | Give every arm the same wider search |
| Confirm | Best two candidates, seeds 1337 and 7331 | 67,108,864 | Select by mean terminal validation loss |
| Horizon | Selected candidate, fresh seed 9001 | 268,435,456 | Check longer stability, loss targets and cost |

The extra rates are one third of the initial minimum and three times its
maximum. Every arm receives this allowance, even when its screen winner is
interior. Candidate budgets are equal in training tokens and seed counts;
they are not equal in GPU hours. Expensive optimizers incur their real cost.

AdamW completes screening and confirmation first. Its settings then become
the fixed fallback for Muon, Shampoo and K-FAC. The advanced arms receive
the same search/confirmation budget as AdamW. All runs start from scratch;
short runs are not extended after their cosine schedule has decayed.

Initial learning-rate grids, before the wider bracket:

| Optimizer | Rates |
|---|---|
| AdamW, Transformer | 0.0001, 0.0003, 0.001, 0.003 |
| AdamW, Mamba | 0.0003, 0.001, 0.003, 0.006 |
| Muon | 0.005, 0.01, 0.02, 0.04 |
| Shampoo | 0.001, 0.003, 0.01, 0.03 |
| K-FAC | 0.0003, 0.001, 0.003, 0.01 |

These neighborhoods include the successful pilot settings. They are not
assertions that the optima lie inside them. Recipe variants are:

| Optimizer | Recipe 1 | Recipe 2 | Recipe 3 |
|---|---|---|---|
| AdamW | beta2 0.95, decay 0.1 | beta2 0.99, decay 0.1 | beta2 0.95, decay 0.01 |
| Muon | momentum 0.9 | momentum 0.95 | momentum 0.99 |
| Shampoo | no graft, momentum 0, refresh 10 | no graft, momentum 0.9, refresh 50 | AdaGrad graft, momentum 0.9, refresh 50 |
| K-FAC | damping 0.0001, refresh 10 | damping 0.001, refresh 10 | damping 0.01, refresh 50 |

Shampoo recipe 2 uses one tenth of the listed rates because its installed
momentum buffer is an unnormalized sum. This compensates for its approximate
steady-state factor of ten; the LR sweep still decides the actual rate.

These are **recipe comparisons**, not isolated causal ablations of every
knob. Some recipes change two settings together. A refresh-50 winner does
not by itself prove that a slower refresh is better independently of damping
or momentum. We retain the exact winning recipe, not individual knobs
assembled from different winning runs.

A candidate must complete every assigned seed with finite terminal
validation. Known non-finite numerical failures count as unsuccessful
candidates and retain their logs. Permission, import, storage, GPU occupancy
and unknown failures stop calibration rather than being scored as optimizer
quality. A confirmed winner at the outer LR boundary blocks main readiness;
it needs a revised, documented search. Neither a timeout nor a partial seed
set can select a winner. Ties use the candidate identifier deterministically.

## Settings held fixed, and why

The search is finite. "Ready for the main study" means that every setting has
a recorded value and selection rule, not that every possible setting has
been optimized. These controls preserve the architecture/optimizer question:

| Setting | Fixed value and reason |
|---|---|
| Architecture | Current parameter-matched definitions in models.py; changing model design would introduce another study factor |
| Tokenizer | Frozen 50,000-token summer tokenizer and checksum; its original training corpus is incompletely documented |
| Data | Pinned FineWeb-Edu sample-10BT revision; existing immutable 512 Mi-token artifact |
| Splits | Document-level normalized-text hash, 98/1/1; equivalent exact text stays in one split; no claim of near-duplicate removal |
| Context | 2048 packed tokens, EOS between documents, next-token cross-entropy in nats |
| Batch | Microbatch 1, accumulation 16, 32,768 tokens/update; full-size pilots demonstrated fit for every arm |
| Initialization/data order | Shared model and sampler seed within paired comparisons; separate Fisher RNG |
| Numerical precision | BF16 autocast, FP32 parameters/state, TF32 off, reference optimizer backend |
| Execution | Eager, activation checkpointing on, same unfused Mamba projection path for all arms so K-FAC hooks execute |
| Schedule | Linear warmup over 3% of tokens, at least two updates; cosine decay to 10% of peak |
| Clipping | Global raw-gradient norm capped at 1.0; pre-clip norm logged; no additional K-FAC update clipping |
| Validation | Same first 262,144 validation tokens, every 128 updates and at completion; test untouched during calibration |
| Checkpoint/log cadence | Checkpoint every 128 updates, console every 10; per-update JSONL metrics |
| Repeated data | Disabled; final budget must fit the prepared corpus |
| Matrix routing | All non-head Linear weights; identical named partition across every arm; tied weights excluded |
| Decay exclusions | Biases, vectors and parameters marked no-weight-decay receive zero decay |
| Advanced matrix decay | 0.1; fallback decay comes from the selected AdamW control |
| AdamW constants | beta1 0.9, epsilon 1e-8, unfused/non-foreach; beta2/decay/LR selected above |
| Muon constants | Nesterov on, five Newton-Schulz steps, spectral shape scaling; momentum/LR selected above |
| Shampoo constants | Factor EMA beta2 0.999, damping epsilon 1e-8, FP64 inverse fourth roots, maximum factor dimension 8192, immediate preconditioning |
| K-FAC constants | Sampled Fisher, factor EMA 0.95, pi-split damping, momentum 0.9; factor and inverse refresh share the selected interval |
| Final seeds | 101, 202, 303, disjoint from all calibration seeds; three is a practical descriptive replication budget, not a power guarantee |

Warmup, clipping and batch are explicit controls supported by working pilots,
not empirically optimal settings. If longer trials show clipping-dominated
training or instability, those observations require a documented revision,
not an unrecorded change while other trials run. Hyperparameter conclusions
are conditional on this tokenizer, model, corpus and training budget.

## Main-study budget and targets

All arms receive the same horizon tokens. Once those trials finish, compute
the improvement from the halfway validation point to terminal validation
for every arm. If the median improvement is at least 0.05 nats, propose a
common final budget of 536,870,912 tokens; otherwise propose 268,435,456.
This predeclared rule chooses a feasible budget within the existing corpus.
It does not establish saturation, asymptotic quality or compute-optimal
training. A larger study remains possible after preparing and recalibrating
on a larger immutable artifact.

Within each architecture/size, the proposed shared loss target is the worst
terminal horizon loss across optimizers, rounded upward to the next tenth.
This gives a threshold every optimizer has demonstrably reached in a pilot.
It is frozen before final seeds, and is a secondary metric. The primary
results remain complete validation-loss curves versus tokens and training
time. Loss 2.5 is not imposed. Targets missed in a final seed are reported
as not reached, rather than extrapolated crossings.

Training time includes optimizer work, Fisher passes, decompositions, data
transfer and in-update compilation. Validation and checkpoint time are
reported separately; active trainer elapsed time is also retained. Projected
GPU hours for 256 Mi, 512 Mi and 1 Gi tokens use measured active time and all
three final seeds. They exclude queue delays and downtime and are estimates,
not promises. Scratch I/O and other users can still influence wall time.

Final reporting uses paired seed-wise differences from AdamW as well as
per-arm mean and standard deviation. Three seeds support descriptive
uncertainty, not a strong significance claim. Preserve unsuccessful trials,
all tuning costs, routing counts, code/data hashes and selected recipes.
Final test evaluation occurs once at the end of each frozen final run;
test losses never select hyperparameters.

## Scope relative to existing work

[Muon Meets Mamba](https://arxiv.org/abs/2608.03941) already studies where to
assign Muon within Mamba-2 and keeps other parameters on AdamW. Our study
does not claim the first use of Muon with Mamba. The proposed contribution
is a controlled comparison including Shampoo and sampled-Fisher K-FAC,
across two architectures and two parameter scales, with measured training
time, common parameter routing and disclosed calibration costs. The
ungrafted versus grafted Shampoo distinction must remain visible in the
eventual optimizer labels and conclusions.

## Runtime, retention and operation

There are 288 screening/bracketing runs, 64 confirmation runs and 16 horizon
runs if every stage finishes. That is 13,421,772,800 training tokens across
calibration, not one model's budget. Pilot-based training-time estimates
are about 260 GPU hours before accounting for the faster refresh recipes
and additional validation/checkpoint overhead. Two GPUs imply roughly five
to six days; the deadline can pause the campaign before it finishes.

The campaign is capped at 120 hours from its first launch and must stop by
October 2 at 00:00 Pacific, ahead of the announced October 3-4 downtime.
Both limits are durable across restarts. It uses at most two GPU UUIDs,
checks occupancy before every trial and takes process locks. It neither
reserves the server nor prevents another user from starting a job.

Completed losing candidates release their checkpoint files after stage
selection. Their metrics, configurations, source identities and failure logs
remain. Shortlisted and selected checkpoints remain; active and paused
checkpoints are never pruned. The original 16 pilot directories are untouched.
Artifacts remain on scratch and need copying to durable project storage
before any server cleanup; checkpoint recovery is not an off-server backup.

On ml2, from this repository after sourcing `../env.sh`:

```bash
bash launch_study.sh --calibrate calibration.json \
  GPU-09f93eaf-a6fa-9651-59fb-b7ddc3656f3a \
  GPU-45699d16-5721-64f5-3624-c657bbbec7e4
cat results/calibration/status.json
tail -n 20 results/calibration-queue.log
tmux attach -t optimizer-training
```

Detach with Ctrl+A then d. Ctrl+C requests a checkpoint and pauses the
campaign. Before its deadline, the same launch command resumes it. An expired
deadline does not reset on relaunch; schedule continuation explicitly after
the server maintenance rather than changing the frozen specification.

The campaign writes stage plans/rankings and `selected.json` as it progresses.
At completion it writes `horizon-summary.csv` and
`main-study-readiness.json`. If no boundary or failure blocks readiness,
it also writes a proposed `final-plan.json` with three fresh seeds. This
plan is reviewable and executable with the ordinary study launcher; creating
it does not start the main comparison.

```bash
.venv/bin/python report.py --results results/calibration/runs \
  --output results/calibration/report
```

Charts and CSVs label optimizer recipes separately. Do not combine screening,
confirmation and horizon losses as repeated seeds of the same experiment.
