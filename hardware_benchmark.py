"""Short GPU throughput screen. Synthetic tokens; no convergence claims."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from models import GPT, Mamba2LM
from pytorch_opt import KFAC, Muon, Shampoo, SOAP, matrix_param_groups, ops

SIZES = {"150m": (768, 16, 30), "300m": (1024, 20, 38)}
OPTIMIZERS = ("adamw", "muon", "shampoo", "kfac")


def model_for(arch, size, sequence, tiny=False):
    width, transformer_layers, mamba_layers = (64, 1, 1) if tiny else SIZES[size]
    vocab = 64 if tiny else 50000
    if arch == "transformer":
        return GPT(vocab, width, width // 64, transformer_layers, 4 * width,
                   sequence, 0.0, True)
    return Mamba2LM(vocab, width, mamba_layers, 16 if tiny else 128,
                    16 if tiny else 64, 2, 4, 16 if tiny else 256, True)


def optimizer_for(model, name, refresh):
    selected = [n for n, m in model.named_modules()
                if isinstance(m, nn.Linear) and n != "lm_head"]
    groups = matrix_param_groups(model, selected, lr=1e-4, adamw_lr=1e-4,
                                 weight_decay=0.1, adamw_wd=0.1)
    if name == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8,
                                 foreach=False, fused=False)
    if name == "muon":
        return Muon(groups)
    if name == "shampoo":
        return Shampoo(groups, max_preconditioner_dim=8192,
                       precondition_frequency=refresh, root_dtype=torch.float64)
    if name == "soap":
        return SOAP(groups, max_preconditioner_dim=8192,
                    precondition_frequency=refresh)
    return KFAC(model, params=groups, fisher_mode="sampled",
                stats_every=refresh, inv_every=refresh)


def summarize(samples, refresh, tokens):
    assert len(samples) % refresh == 0
    seconds = sum(s["seconds"] for s in samples)
    cycles = [statistics.mean(s["seconds"] for s in samples[i:i + refresh])
              for i in range(0, len(samples), refresh)]
    return dict(mean_step_seconds=seconds / len(samples),
                tokens_per_second=tokens * len(samples) / seconds,
                mean_optimizer_seconds=statistics.mean(s["optimizer_seconds"] for s in samples),
                mean_sampled_fisher_seconds=statistics.mean(s["fisher_seconds"] for s in samples),
                cycle_mean_seconds=cycles)


def metadata():
    props = torch.cuda.get_device_properties(0)
    packages = {}
    for name in ("torch", "triton", "mamba-ssm", "causal-conv1d", "pytorch-opt"):
        packages[name] = importlib.metadata.version(name)
    smi = subprocess.run([
        "nvidia-smi", "--query-gpu=name,uuid,memory.total,power.limit,driver_version",
        "--format=csv,noheader"], capture_output=True, text=True, check=False)
    import pytorch_opt
    digest = hashlib.sha256()
    for path in sorted(Path(pytorch_opt.__file__).parent.rglob("*.py")):
        digest.update(path.relative_to(Path(pytorch_opt.__file__).parent).as_posix().encode())
        digest.update(path.read_bytes())
    return dict(gpu=props.name, vram_bytes=props.total_memory,
                capability=list(torch.cuda.get_device_capability(0)),
                cuda=torch.version.cuda, packages=packages, nvidia_smi=smi.stdout.strip(),
                optimizer_source_sha256=digest.hexdigest(),
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                models_sha256=hashlib.sha256(Path(__file__).with_name("models.py").read_bytes()).hexdigest())


def run_case(args):
    arch, size, name = args.worker
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.set_num_threads(4)
    torch.manual_seed(3619)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ops.set_backend("reference")
    model = model_for(arch, size, args.sequence, args.tiny).cuda().train()
    opt = optimizer_for(model, name, args.refresh)
    vocab = model.tok_emb.num_embeddings
    # CPU generation makes identical inputs across GPU architectures.
    data = torch.randint(vocab, (args.accumulation, args.batch, args.sequence + 1)).cuda()
    inputs, targets = data[..., :-1].contiguous(), data[..., 1:].contiguous()
    n_params = sum(p.numel() for p in model.parameters())
    n_selected = sum(p.numel() for g in opt.param_groups if g["use_preconditioner"] for p in g["params"])
    steps = args.refresh * (1 + args.cycles)
    samples = []
    torch.cuda.reset_peak_memory_stats()
    print(f"{arch}/{size}/{name}: {n_params:,} parameters, {steps} steps", flush=True)
    for step in range(steps):
        fisher_events = []
        update_start, update_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        losses = []
        for x, y in zip(inputs, targets):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            if isinstance(opt, KFAC) and step % args.refresh == 0:
                a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                opt.update_curvature(logits)
                b.record()
                fisher_events.append((a, b))
            (loss / args.accumulation).backward()
            losses.append(loss.detach())
            del logits, loss
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        update_start.record()
        opt.step()
        update_end.record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        loss_value = torch.stack(losses).mean().item()
        if not torch.isfinite(torch.tensor(loss_value)):
            raise RuntimeError("Non-finite loss")
        sample = dict(step=step, seconds=elapsed, loss=loss_value,
                      optimizer_seconds=update_start.elapsed_time(update_end) / 1000,
                      fisher_seconds=sum(a.elapsed_time(b) for a, b in fisher_events) / 1000)
        if step >= args.refresh:
            samples.append(sample)
        if (step + 1) % args.refresh == 0:
            print(f"  {step + 1}/{steps} steps; last step {elapsed:.3f}s", flush=True)
    result = dict(status="ok", architecture=arch, size=size, optimizer=name,
                  parameters=n_params, preconditioned_parameters=n_selected,
                  config=dict(sequence=args.sequence, batch=args.batch, accumulation=args.accumulation,
                              refresh=args.refresh, cycles=args.cycles, seed=3619,
                              precision="BF16 autocast / FP32 weights / FP64 matrix decompositions",
                              checkpointing=True, compile=False, mamba_mem_eff_path=False,
                              optimizer_backend="reference", synthetic_tokens=True, tiny=args.tiny),
                  hardware=metadata(), samples=samples,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  **summarize(samples, args.refresh, args.batch * args.sequence * args.accumulation))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"  {result['tokens_per_second']:,.0f} tokens/s; {result['peak_allocated_gib']:.2f} GiB", flush=True)


def run_suite(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    failed = False
    for size in args.sizes:
        for arch in args.architectures:
            for name in args.optimizers:
                filename = output / f"{arch}-{size}-{name}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", arch, size, name,
                           "--output", str(filename)]
                for option in ("sequence", "batch", "accumulation", "refresh", "cycles"):
                    command.extend(["--" + option, str(getattr(args, option))])
                if args.tiny:
                    command.append("--tiny")
                try:
                    subprocess.run(command, check=True, timeout=args.case_timeout)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                    failed = True
                    status = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "error"
                    filename.write_text(json.dumps(dict(status=status, architecture=arch, size=size,
                                                        optimizer=name, detail=str(exc)), indent=2) + "\n")
                    print(f"{arch}/{size}/{name}: {status}, continuing", flush=True)
    print(f"Results: {output.resolve()}", flush=True)
    if failed:
        raise SystemExit("One or more cases failed. Inspect the result JSON and log before proceeding.")


def self_test():
    import contextlib
    import io
    import tempfile
    from unittest.mock import patch

    # Checks the repeated-refresh average rather than hiding expensive steps in a median.
    result = summarize([dict(seconds=t, optimizer_seconds=0, fisher_seconds=0)
                        for t in (9, 1, 9, 1, 9, 1)], 2, 10)
    assert result["tokens_per_second"] == 2 and result["cycle_mean_seconds"] == [5, 5, 5]
    for name in OPTIMIZERS:
        torch.manual_seed(3619)
        model = model_for("transformer", "150m", 8, tiny=True)
        opt = optimizer_for(model, name, 2)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            out = model(torch.randint(64, (1, 8)))
            if isinstance(opt, KFAC):
                opt.update_curvature(out)
            F.cross_entropy(out.flatten(0, 1), torch.randint(64, (8,))).backward()
            opt.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if isinstance(opt, KFAC):
            opt.tracker.remove()
    # A failed worker must leave its diagnostic and fail the overall command.
    with tempfile.TemporaryDirectory() as directory:
        args = argparse.Namespace(output=str(Path(directory) / "results"),
                                  sizes=["150m"], architectures=["transformer"], optimizers=["adamw"],
                                  sequence=8, batch=1, accumulation=1, refresh=2, cycles=1,
                                  tiny=True, case_timeout=1)
        with contextlib.redirect_stdout(io.StringIO()), patch(
                "subprocess.run", side_effect=subprocess.CalledProcessError(1, "worker")):
            try:
                run_suite(args)
            except SystemExit as exc:
                assert exc.code
            else:
                raise AssertionError("Failed benchmark worker was reported as success")
        assert json.loads((Path(args.output) / "transformer-150m-adamw.json").read_text())["status"] == "error"
    print("CPU self-check passed: model routing, optimizer steps and cycle timing summary.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    parser.add_argument("--architectures", nargs="+", choices=("transformer", "mamba"), default=["transformer", "mamba"])
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--optimizers", nargs="+", choices=(*OPTIMIZERS, "soap"), default=list(OPTIMIZERS))
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--refresh", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--case-timeout", type=int, default=180)
    parser.add_argument("--worker", nargs=3, metavar=("ARCH", "SIZE", "OPTIMIZER"), help=argparse.SUPPRESS)
    parser.add_argument("--tiny", action="store_true", help="Small models for script verification only")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if min(args.sequence, args.batch, args.accumulation, args.refresh, args.cycles, args.case_timeout) < 1:
        parser.error("Counts and timeouts must be positive")
    if args.self_test:
        torch.set_num_threads(1)
        self_test()
    elif args.worker:
        run_case(args)
    else:
        run_suite(args)


if __name__ == "__main__":
    main()
