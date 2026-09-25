"""Create, run, and select reproducible optimizer-study trials."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time

from artifacts import atomic_json, file_lock, fingerprint
from optimizers import OPTIMIZERS
from prepare_data import verify_data
from train import Config, source_identity


def load_json(path):
    return json.loads(Path(path).read_text())


def arm_key(architecture, size, optimizer):
    return f"{architecture}/{size}/{optimizer}"


def rate_for(rates, optimizer, architecture):
    value = rates[optimizer]
    return value[architecture] if isinstance(value, dict) else value


def make_plan(spec, phase, data, *, architectures=None, sizes=None, tokens=None, seeds=None,
              selections=None):
    if phase not in ("pilot", "tune-adamw", "tune-others", "final"):
        raise ValueError("Unknown phase")
    architectures = architectures or spec["architectures"]
    sizes = sizes or spec["sizes"]
    optimizers = spec["optimizers"]
    if len(optimizers) != 4 or set(optimizers[:3]) != {"adamw", "muon", "shampoo"} or (
            optimizers[3] not in ("kfac", "soap")):
        raise ValueError("Use AdamW, Muon, Shampoo, and exactly one of K-FAC/SOAP")
    stage = spec["pilot" if phase == "pilot" else "final" if phase == "final" else "tuning"]
    budget = tokens if tokens is not None else stage["tokens"]
    if budget is None:
        raise ValueError("Choose --tokens for final runs after reviewing pilots; no fixed main-study budget")
    seeds = seeds or stage["seeds"]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be distinct and nonempty")
    if phase == "final" and len(seeds) < 2:
        raise ValueError("Final comparisons require repeated seeds")
    manifest = verify_data(data, checksums=False)
    sources = source_identity()
    if phase in ("tune-others", "final"):
        if selections is None:
            raise ValueError("This phase requires --selection from completed learning-rate trials")
        if selections["data_id"] != manifest["data_id"] or selections["sources"] != sources:
            raise ValueError("LR selection used different data or training/optimizer code")
        if phase == "final" and any(set(seeds) & set(w["seeds"]) for w in selections["winners"].values()):
            raise ValueError("Final seeds must differ from learning-rate tuning seeds")
    chosen = ["adamw"] if phase == "tune-adamw" else [o for o in optimizers if o != "adamw"] if phase == "tune-others" else optimizers
    trials = []
    for architecture in architectures:
        for size in sizes:
            baseline_key = arm_key(architecture, size, "adamw")
            if phase in ("tune-others", "final"):
                fallback = selections["winners"][baseline_key]["lr"]
            else:
                fallback = rate_for(spec["pilot"]["learning_rates"], "adamw", architecture)
            for optimizer in chosen:
                if phase == "pilot":
                    rates = [rate_for(stage["learning_rates"], optimizer, architecture)]
                elif phase == "final":
                    winner = selections["winners"][arm_key(architecture, size, optimizer)]
                    rates = [winner["lr"]]
                    if optimizer != "adamw" and winner["fallback_lr"] != fallback:
                        raise ValueError("Selected primary and fallback learning rates are inconsistent")
                else:
                    rates = rate_for(stage["learning_rates"], optimizer, architecture)
                    reference_count = len(rate_for(stage["learning_rates"], "adamw", architecture))
                    if len(rates) != reference_count or len(set(rates)) != len(rates):
                        raise ValueError("Give every optimizer the same number of distinct LR candidates")
                for lr in rates:
                    for seed in seeds:
                        values = dict(spec["training"], architecture=architecture, size=size,
                                      optimizer=optimizer, phase="tune" if phase.startswith("tune") else phase,
                                      total_tokens=budget, seed=seed, data_seed=seed, lr=lr,
                                      fallback_lr=lr if optimizer == "adamw" else fallback,
                                      evaluate_test=phase == "final")
                        config = Config(**values)
                        warmup_steps = max(spec["minimum_warmup_steps"],
                                           round(budget * spec["warmup_fraction"] / config.tokens_per_step))
                        config.warmup_tokens = warmup_steps * config.tokens_per_step
                        config.validate()
                        available = (manifest["splits"]["train"]["tokens"] - 1) // config.sequence * config.sequence
                        if not config.allow_repeated_data and config.total_tokens > available:
                            raise ValueError("Prepare enough data for the final budget before tuning learning rates")
                        if any(manifest["splits"][split]["tokens"] <= config.eval_tokens
                               for split in (("val", "test") if config.evaluate_test else ("val",))):
                            raise ValueError("Prepared holdouts are smaller than the evaluation budget")
                        config_dict = asdict(config)
                        trial_id = fingerprint(dict(config=config_dict, data_id=manifest["data_id"], sources=sources))[:12]
                        trials.append(dict(id=f"{architecture}-{size}-{optimizer}-s{seed}-{trial_id}", config=config_dict))
    random.Random(spec["plan_seed"]).shuffle(trials)
    plan = dict(version=1, phase=phase, data=str(Path(data).resolve()), data_id=manifest["data_id"],
                sources=sources, trials=trials, selection=selections,
                created_at=datetime.now(timezone.utc).isoformat())
    plan["plan_id"] = fingerprint(plan)
    return plan


def check_plan(plan):
    value = dict(plan)
    plan_id = value.pop("plan_id")
    if fingerprint(value) != plan_id or len({t["id"] for t in plan["trials"]}) != len(plan["trials"]):
        raise ValueError("Plan fingerprint or trial IDs are invalid")
    for trial in plan["trials"]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", trial["id"]):
            raise ValueError("Unsafe trial ID")
        Config(**trial["config"]).validate()


def ensure_free_gpu(gpu):
    if gpu == "cpu":
        return
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", gpu):
        raise ValueError("Select a full GPU UUID from nvidia-smi")
    subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"],
                   check=True, capture_output=True, text=True, timeout=30)
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True, timeout=30)
    if any(line.split(",")[0].strip() == gpu for line in result.stdout.splitlines()):
        raise RuntimeError(f"{gpu} has an active compute process; leaving pending trials untouched")


def run_plan(plan_path, results, gpus, *, hours=None, retry_failed=False):
    plan = load_json(plan_path)
    check_plan(plan)
    if plan["sources"] != source_identity() or verify_data(plan["data"])["data_id"] != plan["data_id"]:
        raise ValueError("The plan's source code or dataset changed; generate a new plan")
    if not 1 <= len(gpus) <= 2 or len(set(gpus)) != len(gpus):
        raise ValueError("Select one or two distinct GPUs")
    if hours is not None and (not math.isfinite(hours) or hours <= 0):
        raise ValueError("hours must be finite and positive")
    devices = {t["config"]["device"] for t in plan["trials"]}
    if devices != ({"cpu"} if gpus == ["cpu"] else {"cuda"}) or ("cpu" in gpus and gpus != ["cpu"]):
        raise ValueError("Plan device and selected GPUs disagree")
    for gpu in gpus:
        ensure_free_gpu(gpu)
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    for trial in plan["trials"]:
        pending.put(trial)
    stopping = threading.Event()
    failures = []
    deadline = time.monotonic() + hours * 3600 if hours else math.inf

    def stop(signum, frame):
        stopping.set()
        print("Stopping queue; active trainers will finish their update and checkpoint.", flush=True)

    old_signals = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}

    def worker(gpu):
        lock_path = Path(tempfile.gettempdir()) / f"optimizer-study-{os.getuid()}-{gpu}.lock"
        try:
            with file_lock(lock_path):
                while not stopping.is_set() and time.monotonic() < deadline:
                    try:
                        trial = pending.get_nowait()
                    except queue.Empty:
                        break
                    directory = results / trial["id"]
                    directory.mkdir(exist_ok=True)
                    status_path = directory / "status.json"
                    status = load_json(status_path) if status_path.exists() else {}
                    if status.get("status") == "complete":
                        if load_json(directory / "config.json") != trial["config"]:
                            raise ValueError("Completed run config differs from plan")
                        identity = load_json(directory / "metadata.json")["identity"]
                        if identity != dict(config_id=fingerprint(trial["config"]), data_id=plan["data_id"], sources=plan["sources"]):
                            raise ValueError("Completed run identity differs from plan")
                        print(f"Already complete: {trial['id']}", flush=True)
                        continue
                    if status.get("status") == "failed" and not retry_failed:
                        failures.append(trial["id"])
                        print(f"Failed run retained: {trial['id']}; use --retry-failed only after diagnosis", flush=True)
                        continue
                    ensure_free_gpu(gpu)
                    config_path = directory / "requested-config.json"
                    atomic_json(config_path, trial["config"])
                    command = [sys.executable, str(Path(__file__).with_name("train.py")), "--config", str(config_path),
                               "--data", plan["data"], "--output", str(directory)]
                    if (directory / "latest.json").exists():
                        command.append("--resume")
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu == "cpu" else gpu)
                    print(f"Starting {trial['id']} on {gpu}; log: {directory / 'console.log'}", flush=True)
                    started = time.monotonic()
                    terminate_at = None
                    with open(directory / "console.log", "a", buffering=1) as stdout, open(directory / "stderr.log", "a", buffering=1) as stderr:
                        process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr)
                        while process.poll() is None:
                            if stopping.is_set() or time.monotonic() >= deadline:
                                stopping.set()
                                if terminate_at is None:
                                    process.terminate()
                                    terminate_at = time.monotonic()
                                elif time.monotonic() - terminate_at > 600:
                                    process.kill()
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                pass
                    execution = dict(returncode=process.returncode, seconds=time.monotonic() - started,
                                     gpu=gpu, ended_at=datetime.now(timezone.utc).isoformat(), command=command)
                    atomic_json(directory / "execution.json", execution)
                    with open(directory / "attempts.jsonl", "a") as attempts:
                        attempts.write(json.dumps(execution) + "\n")
                    status = load_json(status_path) if status_path.exists() else {}
                    if process.returncode or status.get("status") not in ("complete", "paused"):
                        failures.append(trial["id"])
                        if status.get("status") not in ("failed", "complete"):
                            atomic_json(status_path, dict(status="failed", error=f"Trainer exit {process.returncode}; inspect stderr.log"))
                        print(f"FAILED {trial['id']}; inspect {directory / 'stderr.log'}", flush=True)
                    elif status["status"] == "paused":
                        stopping.set()
                        print(f"PAUSED {trial['id']}; rerun the same queue command to resume", flush=True)
                    else:
                        print(f"COMPLETE {trial['id']}", flush=True)
        except Exception as exc:
            stopping.set()
            failures.append(f"{gpu}: {exc}")
            print(f"Queue stopped: {gpu}: {exc}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(worker, gpus))
    finally:
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)
    if failures:
        raise RuntimeError("Failed trials or worker errors: " + "; ".join(failures))
    print("Queue paused; rerun the same command to continue." if stopping.is_set() or not pending.empty()
          else "Queue complete.", flush=True)


def select(plan_paths, results):
    groups = {}
    data_id = sources = None
    inherited = {}
    protocols = {}
    platforms = set()
    for plan_path in plan_paths:
        plan = load_json(plan_path)
        check_plan(plan)
        if plan["phase"] not in ("tune-adamw", "tune-others"):
            raise ValueError("Select learning rates using tuning plans, never pilots or final/test runs")
        if data_id is not None and (data_id != plan["data_id"] or sources != plan["sources"]):
            raise ValueError("Tuning plans use different data/code")
        data_id, sources = plan["data_id"], plan["sources"]
        if plan["selection"]:
            inherited.update(plan["selection"]["winners"])
        for trial in plan["trials"]:
            config = trial["config"]
            # Only LR and paired seeds may vary within an architecture/size search.
            protocol = {k: v for k, v in config.items() if k not in
                        ("lr", "fallback_lr", "optimizer", "seed", "data_seed")}
            comparison = (config["architecture"], config["size"])
            if protocols.setdefault(comparison, protocol) != protocol:
                raise ValueError("LR candidates have different training budgets or protocols")
            directory = Path(results) / trial["id"]
            status_path = directory / "status.json"
            if not status_path.exists():
                raise ValueError(f"Trial has not finished: {trial['id']}")
            status = load_json(status_path)
            if status["status"] not in ("complete", "failed"):
                raise ValueError(f"Trial is still {status['status']}: {trial['id']}")
            key = arm_key(config["architecture"], config["size"], config["optimizer"])
            candidate = groups.setdefault(key, {}).setdefault(str(config["lr"]), [])
            loss = None
            if status["status"] == "complete":
                if load_json(directory / "config.json") != config or status["tokens"] != config["total_tokens"]:
                    raise ValueError("Trial configuration or completed token budget differs")
                metadata = load_json(directory / "metadata.json")
                if metadata["identity"]["data_id"] != data_id or metadata["identity"]["sources"] != sources:
                    raise ValueError("Trial data/code identity differs")
                platforms.add(fingerprint({k: metadata["hardware"].get(k) for k in
                                           ("device", "name", "capability", "torch", "cuda", "versions")}))
                if len(platforms) > 1:
                    raise ValueError("Do not combine tuning trials from different hardware/software stacks")
                evaluations = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
                terminal = [r for r in evaluations if r["kind"] == "val" and r["tokens"] == config["total_tokens"]]
                if not terminal or not math.isfinite(terminal[-1]["loss"]):
                    raise ValueError("Completed trial has no finite terminal validation loss")
                loss = terminal[-1]["loss"]
            candidate.append(dict(seed=config["seed"], loss=loss, trial=trial["id"], config=config))
    winners = dict(inherited)
    candidates = {}
    for key, rates in groups.items():
        seed_sets = [{r["seed"] for r in runs} for runs in rates.values()]
        if any(seeds != seed_sets[0] for seeds in seed_sets) or any(
                len(runs) != len({r["seed"] for r in runs}) for runs in rates.values()):
            raise ValueError("LR candidates have unequal or duplicate seed assignments")
        if not key.endswith("/adamw") and len({r["config"]["fallback_lr"] for runs in rates.values() for r in runs}) != 1:
            raise ValueError("All advanced-optimizer LR candidates must share the fixed fallback LR")
        summaries = []
        for rate, runs in rates.items():
            valid = all(r["loss"] is not None for r in runs)
            summaries.append(dict(lr=float(rate), valid=valid,
                                  mean_val_loss=statistics.mean(r["loss"] for r in runs) if valid else None,
                                  fallback_lr=runs[0]["config"]["fallback_lr"],
                                  tokens=runs[0]["config"]["total_tokens"], seeds=sorted(seed_sets[0]),
                                  trials=[r["trial"] for r in runs]))
        eligible = [r for r in summaries if r["valid"]]
        if not eligible:
            raise ValueError(f"No LR completed all seeds for {key}; inspect failures before changing protocol")
        winner = min(eligible, key=lambda r: (r["mean_val_loss"], r["lr"]))
        winner = dict(winner, boundary=winner["lr"] in (min(float(r) for r in rates), max(float(r) for r in rates)))
        if winner["boundary"]:
            print(f"LR optimum is at a grid boundary for {key}; consider expanding every affected arm's search.", flush=True)
        winners[key], candidates[key] = winner, summaries
    return dict(version=1, data_id=data_id, sources=sources, winners=winners, candidates=candidates,
                plans=[str(Path(p).resolve()) for p in plan_paths])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--spec", default="study.json")
    p.add_argument("--phase", choices=("pilot", "tune-adamw", "tune-others", "final"), required=True)
    p.add_argument("--data")
    p.add_argument("--architectures", nargs="+", choices=("transformer", "mamba"))
    p.add_argument("--sizes", nargs="+", choices=("150m", "300m"))
    p.add_argument("--tokens", type=int)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--selection")
    p.add_argument("--output", required=True)
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--results", default="results/training")
    p.add_argument("--gpus", nargs="+", required=True)
    p.add_argument("--hours", type=float)
    p.add_argument("--retry-failed", action="store_true")
    p = sub.add_parser("select")
    p.add_argument("--plans", nargs="+", required=True)
    p.add_argument("--results", default="results/training")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_plan(args.plan, args.results, args.gpus, hours=args.hours, retry_failed=args.retry_failed)
        return
    if args.command == "select":
        value = select(args.plans, args.results)
    else:
        spec = load_json(args.spec)
        value = make_plan(spec, args.phase, args.data or spec["data"], architectures=args.architectures,
                          sizes=args.sizes, tokens=args.tokens, seeds=args.seeds,
                          selections=load_json(args.selection) if args.selection else None)
        print(f"{len(value['trials'])} trials; {sum(t['config']['total_tokens'] for t in value['trials']):,} total planned training tokens")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("Output already exists; plans/selections are immutable, choose a new path")
    atomic_json(target, value)
    print(target)


if __name__ == "__main__":
    main()
