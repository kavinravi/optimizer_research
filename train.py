"""Single-GPU causal-LM training with token budgets and complete checkpoints."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from artifacts import atomic_json, file_lock, fingerprint, sha256_file, sync_directory
from models import model_for
from optimizers import KFAC, OPTIMIZERS, optimizer_for
from prepare_data import DATASET, verify_data
from pytorch_opt import ops


@dataclass
class Config:
    architecture: str = "transformer"
    size: str = "150m"
    optimizer: str = "adamw"
    phase: str = "pilot"
    total_tokens: int = 33554432
    sequence: int = 2048
    micro_batch: int = 1
    accumulation: int = 16
    seed: int = 3619
    data_seed: int = 3619
    lr: float = 3e-4
    fallback_lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_tokens: int = 1048576
    min_lr_ratio: float = 0.1
    refresh: int = 10
    damping: float = 1e-3
    selection: str = "all"
    eval_every: int = 64
    eval_tokens: int = 262144
    checkpoint_every: int = 64
    log_every: int = 10
    precision: str = "bf16"
    device: str = "cuda"
    checkpointing: bool = True
    allow_repeated_data: bool = False
    evaluate_test: bool = False
    tiny: bool = False

    @property
    def tokens_per_step(self):
        return self.sequence * self.micro_batch * self.accumulation

    def validate(self):
        if self.architecture not in ("transformer", "mamba") or self.size not in ("150m", "300m"):
            raise ValueError("Use transformer/mamba and 150m/300m")
        if self.optimizer not in (*OPTIMIZERS, "soap"):
            raise ValueError("Unknown optimizer")
        if self.phase not in ("pilot", "tune", "final", "verification"):
            raise ValueError("Unknown study phase")
        for name in ("total_tokens", "sequence", "micro_batch", "accumulation", "refresh",
                     "eval_every", "eval_tokens", "checkpoint_every", "log_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "data_seed", "warmup_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("checkpointing", "allow_repeated_data", "evaluate_test", "tiny"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("lr", "fallback_lr", "damping"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.min_lr_ratio <= 1 or self.warmup_tokens >= self.total_tokens:
            raise ValueError("Invalid learning-rate schedule")
        if self.total_tokens % self.tokens_per_step:
            raise ValueError(f"total_tokens must be divisible by {self.tokens_per_step}")
        if self.eval_tokens % self.sequence:
            raise ValueError("eval_tokens must be divisible by sequence length")
        if self.selection not in ("all", "mamba_output") or (
                self.selection == "mamba_output" and self.architecture != "mamba"):
            raise ValueError("Invalid matrix selection for this architecture")
        if self.optimizer == "adamw" and self.lr != self.fallback_lr:
            raise ValueError("AdamW must use the same peak LR on every group")
        if self.device not in ("cpu", "cuda") or self.precision not in ("fp32", "bf16"):
            raise ValueError("Use cpu/cuda and fp32/bf16")
        if self.device == "cpu" and (self.precision != "fp32" or self.architecture == "mamba"):
            raise ValueError("CPU checks support FP32 Transformer only")
        if self.evaluate_test and self.phase != "final":
            raise ValueError("Held-out test evaluation is only allowed for frozen final runs")
        if self.tiny and self.phase != "verification":
            raise ValueError("Tiny models are only for verification")
        return self


class BlockStream:
    """A shuffled epoch of packed blocks, reproducible from seed and cursor."""
    def __init__(self, path, sequence, seed):
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        self.sequence = sequence
        self.seed = seed
        self.blocks = (len(self.tokens) - 1) // sequence
        if self.blocks < 1:
            raise ValueError(f"Too few tokens in {path}")
        self.position = 0
        self._epoch = None
        self._order = None

    def batch(self, count):
        windows = []
        for _ in range(count):
            epoch, index = divmod(self.position, self.blocks)
            if self._epoch != epoch:
                generator = torch.Generator().manual_seed((self.seed + epoch) % (2**63))
                self._order = torch.randperm(self.blocks, generator=generator).numpy()
                self._epoch = epoch
            start = int(self._order[index]) * self.sequence
            windows.append(self.tokens[start:start + self.sequence + 1])
            self.position += 1
        batch = torch.from_numpy(np.asarray(windows, dtype=np.int64))
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()

    def state_dict(self):
        return dict(position=self.position, seed=self.seed, sequence=self.sequence, blocks=self.blocks)

    def load_state_dict(self, state):
        if any(state[k] != getattr(self, k) for k in ("seed", "sequence", "blocks")):
            raise ValueError("Checkpoint data sampler does not match this dataset")
        if type(state["position"]) is not int or state["position"] < 0:
            raise ValueError("Invalid data cursor")
        self.position = state["position"]
        self._epoch = self._order = None


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


@contextmanager
def preserved_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def lr_multiplier(config, completed_tokens):
    if config.warmup_tokens and completed_tokens < config.warmup_tokens:
        return completed_tokens / config.warmup_tokens
    progress = min(1.0, (completed_tokens - config.warmup_tokens) /
                   (config.total_tokens - config.warmup_tokens))
    return config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, path, config, device):
    tokens = np.memmap(path, dtype="<u2", mode="r")
    blocks = config.eval_tokens // config.sequence
    if len(tokens) < blocks * config.sequence + 1:
        raise ValueError("Validation/test split is too short for the fixed evaluation budget")
    was_training = model.training
    loss_sum = 0.0
    with preserved_rng():
        model.eval()
        try:
            for start_block in range(0, blocks, config.micro_batch):
                count = min(config.micro_batch, blocks - start_block)
                batch = np.asarray([tokens[i * config.sequence:(i + 1) * config.sequence + 1]
                                    for i in range(start_block, start_block + count)], dtype=np.int64)
                batch = torch.from_numpy(batch).to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.precision == "bf16"):
                    logits = model(batch[:, :-1].contiguous())
                    loss = F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].reshape(-1), reduction="sum")
                loss_sum += loss.item()
        finally:
            model.train(was_training)
    value = loss_sum / config.eval_tokens
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite evaluation loss")
    return value


def source_identity():
    import pytorch_opt
    root = Path(__file__).parent
    files = {name: sha256_file(root / name) for name in
             ("train.py", "models.py", "optimizers.py", "prepare_data.py", "artifacts.py")}
    opt_root = Path(pytorch_opt.__file__).parent
    files["pytorch_opt"] = fingerprint({str(p.relative_to(opt_root)): sha256_file(p)
                                       for p in sorted(opt_root.rglob("*.py"))})
    return files


def hardware_info(device):
    result = dict(device=device.type, torch=torch.__version__, cuda=torch.version.cuda,
                  platform=sys.platform, python=sys.version)
    result["versions"] = {}
    for name in ("numpy", "triton", "mamba-ssm", "causal-conv1d"):
        try:
            result["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["versions"][name] = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(name=props.name, capability=list(torch.cuda.get_device_capability(device)),
                      memory_bytes=props.total_memory, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    return result


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_checkpoint(directory):
    directory = Path(directory)
    commit = json.loads((directory / "latest.json").read_text())
    filename = commit["file"]
    if Path(filename).name != filename or not filename.startswith("step_") or not filename.endswith(".pt"):
        raise ValueError("Invalid checkpoint filename")
    # Checkpoints are local artifacts produced by this trainer, never third-party downloads.
    state = torch.load(directory / filename, map_location="cpu", weights_only=False)
    if state["step"] != commit["step"] or state["tokens"] != commit["tokens"]:
        raise ValueError("Checkpoint and commit record disagree")
    return state, commit


def train(config, data, output, *, resume=False, stop_after_steps=None, max_seconds=None):
    config.validate()
    if stop_after_steps is not None and stop_after_steps < 1:
        raise ValueError("stop_after_steps must be positive")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    output, data = Path(output), Path(data)
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".run.lock"):
        return _train(config, data, output, resume, stop_after_steps, max_seconds)


def _train(config, data, output, resume, stop_after_steps, max_seconds):
    attempt_start = time.perf_counter()
    config_dict = asdict(config)
    config_id = fingerprint(config_dict)
    if (output / "config.json").exists() and not resume:
        raise ValueError("Run directory already exists; use --resume or choose a new run directory")
    if resume and not (output / "latest.json").exists():
        raise ValueError("No committed checkpoint exists in this run directory")
    manifest = verify_data(data)
    if not config.tiny and manifest["recipe"]["dataset"] != DATASET:
        raise ValueError("Study training requires a prepared FineWeb-Edu dataset")
    stream = BlockStream(data / "train.bin", config.sequence, config.data_seed)
    if not config.allow_repeated_data and config.total_tokens // config.sequence > stream.blocks:
        raise ValueError("Token budget exceeds one prepared epoch; prepare more data or explicitly allow repetition")
    if manifest["splits"]["val"]["tokens"] < config.eval_tokens + 1:
        raise ValueError("Validation split does not cover eval_tokens plus one target token")
    if config.evaluate_test and manifest["splits"]["test"]["tokens"] < config.eval_tokens + 1:
        raise ValueError("Test split is too small")
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Select exactly one available GPU with CUDA_VISIBLE_DEVICES=GPU-UUID")
        if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16; choose FP32 explicitly for a separate comparison")
    torch.set_num_threads(4 if not config.tiny else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    ops.set_backend("reference")
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    hardware = hardware_info(device)
    sources = source_identity()
    identity = dict(config_id=config_id, data_id=manifest["data_id"], sources=sources)
    if resume and (output / "status.json").exists():
        done = json.loads((output / "status.json").read_text())
        if done["status"] == "complete":
            original = json.loads((output / "metadata.json").read_text())
            if original["identity"] != identity:
                raise ValueError("Completed run has different config, data or code")
            print(f"Already complete: {output}", flush=True)
            return done
    model = model_for(config.architecture, config.size, config.sequence, config.tiny,
                      vocab_size=manifest["vocab_size"], checkpointing=config.checkpointing)
    initial_sha = model_digest(model)
    model.to(device).train()
    optimizer, routing = optimizer_for(model, config.optimizer, lr=config.lr,
                                       fallback_lr=config.fallback_lr, weight_decay=config.weight_decay,
                                       refresh=config.refresh, selection=config.selection, damping=config.damping)
    fisher_rng = torch.Generator(device=device).manual_seed(config.seed + 104729)
    counters = dict(step=0, tokens=0, train_seconds=0.0, eval_seconds=0.0,
                    checkpoint_seconds=0.0, best_val_loss=None, last_eval_step=-1)
    wall_base = 0.0
    checkpoint_step = -1
    metrics_path = output / "metrics.jsonl"
    if resume:
        state, commit = read_checkpoint(output)
        if state["identity"] != identity or state["config"] != config_dict:
            raise ValueError("Resume requires identical config, data fingerprint and training/optimizer code")
        old_hardware = state["hardware"]
        if any(hardware.get(k) != old_hardware.get(k) for k in ("device", "name", "capability", "torch", "cuda", "versions")):
            raise ValueError("Resume hardware/software changed; keep timing comparisons on the original GPU type and stack")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        stream.load_state_dict(state["sampler"])
        fisher_rng.set_state(state["fisher_rng"].cpu())
        counters = state["counters"]
        counters["checkpoint_seconds"] = commit["checkpoint_seconds"]
        wall_base = commit["wall_seconds"]
        checkpoint_step = counters["step"]
        if counters["tokens"] != counters["step"] * config.tokens_per_step or (
                stream.position * config.sequence != counters["tokens"]):
            raise ValueError("Checkpoint counters/data position are inconsistent")
        if state["scheduler"] != dict(kind="token_warmup_cosine", completed_tokens=counters["tokens"]):
            raise ValueError("Checkpoint scheduler state is inconsistent")
        if not metrics_path.exists() or metrics_path.stat().st_size < commit["metrics_bytes"]:
            raise ValueError("Committed metrics are missing or truncated")
        with open(metrics_path, "r+b") as log:
            log.seek(commit["metrics_bytes"])
            discarded = log.read()
            if discarded:
                (output / f"discarded-{uuid.uuid4().hex}.jsonl").write_bytes(discarded)
                log.truncate(commit["metrics_bytes"])
        restore_rng(state["rng"])
        del state
    else:
        atomic_json(output / "config.json", config_dict)
        atomic_json(output / "data_manifest.json", manifest)
        metadata = dict(identity=identity, hardware=hardware, routing=routing,
                        parameters=sum(p.numel() for p in model.parameters()), initial_model_sha256=initial_sha,
                        started_at=datetime.now(timezone.utc).isoformat(), data_directory=str(data.resolve()),
                        versions={p: importlib.metadata.version(p) for p in ("torch", "numpy", "tokenizers", "pytorch-opt")})
        atomic_json(output / "metadata.json", metadata)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True,
                                capture_output=True, check=True, timeout=60)
        (output / "environment.txt").write_text(freeze.stdout)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; finishing the current update before checkpointing.", flush=True)

    previous_signals = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    log = open(metrics_path, "a", buffering=1)

    def wall_seconds():
        return wall_base + time.perf_counter() - attempt_start

    def metric(kind, **extra):
        row = dict(kind=kind, step=counters["step"], tokens=counters["tokens"],
                   train_seconds=counters["train_seconds"], wall_seconds=wall_seconds(), **extra)
        log.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def validate(split="val"):
        synchronize(device)
        started = time.perf_counter()
        value = evaluate(model, data / f"{split}.bin", config, device)
        synchronize(device)
        elapsed = time.perf_counter() - started
        counters["eval_seconds"] += elapsed
        if split == "val":
            counters["last_eval_step"] = counters["step"]
            best = counters["best_val_loss"]
            counters["best_val_loss"] = value if best is None else min(best, value)
        metric(split, loss=value, eval_tokens=config.eval_tokens, eval_seconds=elapsed)
        print(f"{split} step={counters['step']} tokens={counters['tokens']:,} loss={value:.5f} "
              f"train_seconds={counters['train_seconds']:.1f}", flush=True)
        return value

    def save_checkpoint():
        nonlocal checkpoint_step
        # A pause before another update can reuse an already committed checkpoint.
        if checkpoint_step == counters["step"]:
            return
        synchronize(device)
        started = time.perf_counter()
        if any(not torch.isfinite(p).all().item() for p in model.parameters()):
            raise FloatingPointError("Refusing to checkpoint non-finite model parameters")
        state = dict(version=1, identity=identity, config=config_dict, hardware=hardware,
                     step=counters["step"], tokens=counters["tokens"], counters=dict(counters),
                     model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=stream.state_dict(),
                     rng=rng_state(), fisher_rng=fisher_rng.get_state(),
                     scheduler=dict(kind="token_warmup_cosine", completed_tokens=counters["tokens"]))
        filename = f"step_{counters['step']:08d}.pt"
        # The commit pointer is published last, after the checkpoint and log are durable.
        temporary = output / (filename + ".tmp")
        with open(temporary, "wb") as stream_file:
            torch.save(state, stream_file)
            stream_file.flush()
            os.fsync(stream_file.fileno())
        os.replace(temporary, output / filename)
        sync_directory(output)
        elapsed = time.perf_counter() - started
        counters["checkpoint_seconds"] += elapsed
        metric("checkpoint", seconds=elapsed, file=filename)
        log.flush()
        os.fsync(log.fileno())
        commit = dict(file=filename, step=counters["step"], tokens=counters["tokens"],
                      metrics_bytes=log.tell(), wall_seconds=wall_seconds(),
                      checkpoint_seconds=counters["checkpoint_seconds"])
        previous = json.loads((output / "latest.json").read_text()) if (output / "latest.json").exists() else None
        if previous:
            atomic_json(output / "previous.json", previous)
        atomic_json(output / "latest.json", commit)
        keep = {filename, previous["file"] if previous else filename}
        for path in output.glob("step_*.pt"):
            if path.name not in keep:
                path.unlink()
        checkpoint_step = counters["step"]
        print(f"Checkpoint committed: {output / filename}", flush=True)

    try:
        if resume:
            metric("resume", hardware=hardware)
        else:
            validate()
            save_checkpoint()
        atomic_json(output / "status.json", dict(status="running", step=counters["step"], tokens=counters["tokens"]))
        starting_step = counters["step"]
        while counters["tokens"] < config.total_tokens:
            if stop_requested or (max_seconds is not None and time.perf_counter() - attempt_start >= max_seconds):
                break
            if stop_after_steps is not None and counters["step"] - starting_step >= stop_after_steps:
                break
            synchronize(device)
            started = time.perf_counter()
            multiplier = lr_multiplier(config, counters["tokens"] + config.tokens_per_step)
            for group in optimizer.param_groups:
                group["lr"] = (config.lr if group["use_preconditioner"] else config.fallback_lr) * multiplier
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(config.accumulation):
                x, y = stream.batch(config.micro_batch)
                x, y = x.to(device), y.to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.precision == "bf16"):
                    logits = model(x)
                    loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
                if isinstance(optimizer, KFAC):
                    optimizer.update_curvature(logits, generator=fisher_rng)
                (loss / config.accumulation).backward()
                losses.append(loss.detach())
                del logits, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            elapsed = time.perf_counter() - started
            mean_loss = torch.stack(losses).mean().item()
            if not math.isfinite(mean_loss):
                raise FloatingPointError("Non-finite training loss; no partial update is checkpointed")
            counters["step"] += 1
            counters["tokens"] += config.tokens_per_step
            counters["train_seconds"] += elapsed
            metric("train", loss=mean_loss, step_seconds=elapsed, tokens_per_second=config.tokens_per_step / elapsed,
                   lr=config.lr * multiplier, fallback_lr=config.fallback_lr * multiplier,
                   grad_norm=grad_norm.item(), epoch=stream.position // stream.blocks,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
            if counters["step"] % config.log_every == 0 or counters["step"] == 1:
                print(f"step={counters['step']} tokens={counters['tokens']:,} loss={mean_loss:.5f} "
                      f"tokens/s={config.tokens_per_step / elapsed:,.0f}", flush=True)
            if counters["step"] % config.eval_every == 0 and not stop_requested:
                validate()
            if counters["step"] % config.checkpoint_every == 0 and counters["tokens"] < config.total_tokens:
                save_checkpoint()
        complete = counters["tokens"] == config.total_tokens
        if complete and counters["last_eval_step"] != counters["step"]:
            validate()
        test_loss = None
        if complete and config.evaluate_test:
            # A completed run is handled before re-entering this function by the queue.
            test_loss = validate("test")
        save_checkpoint()
        status = dict(status="complete" if complete else "paused", **counters,
                      wall_seconds=wall_seconds(), test_loss=test_loss, config_id=config_id,
                      data_id=manifest["data_id"])
        atomic_json(output / "status.json", status)
        print(f"{status['status'].upper()}: {output}", flush=True)
        return status
    except BaseException as exc:
        atomic_json(output / "status.json", dict(status="failed", step=counters["step"],
                    tokens=counters["tokens"], error=f"{type(exc).__name__}: {exc}", wall_seconds=wall_seconds()))
        raise
    finally:
        log.close()
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
        if isinstance(optimizer, KFAC):
            optimizer.tracker.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--max-hours", type=float)
    args = parser.parse_args()
    config = Config(**json.loads(Path(args.config).read_text())).validate()
    train(config, args.data, args.output, resume=args.resume,
          stop_after_steps=args.stop_after_steps,
          max_seconds=args.max_hours * 3600 if args.max_hours is not None else None)


if __name__ == "__main__":
    main()
