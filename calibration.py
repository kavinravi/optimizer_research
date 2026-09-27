"""Bounded, resumable calibration. Never launches the final comparison."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import time

from artifacts import atomic_json, file_lock, fingerprint, sha256_file
from prepare_data import verify_data
from report import rows_from, write_csv
from study import arm_key, check_plan, load_json, rate_for, run_plan
from train import Config, source_identity


def candidate_key(config):
    return fingerprint({k: v for k, v in config.items() if k not in (
        "phase", "seed", "data_seed", "total_tokens", "warmup_tokens",
        "evaluate_test", "eval_every", "checkpoint_every", "log_every")})[:16]


def candidates(spec, names, baselines=None, expansion=False):
    values = []
    for architecture in spec["architectures"]:
        for size in spec["sizes"]:
            baseline = baselines[arm_key(architecture, size, "adamw")][0]["config"] if baselines else None
            for name in names:
                rates = rate_for(spec["learning_rates"], name, architecture)
                if expansion:
                    rates = [min(rates) / 3, max(rates) * 3]
                for recipe in spec["recipes"][name]:
                    recipe = dict(recipe)
                    scale = recipe.pop("lr_scale", 1)
                    for rate in rates:
                        config = asdict(Config(**(spec["training"] | recipe | dict(
                            architecture=architecture, size=size, optimizer=name, lr=rate * scale))))
                        if name == "adamw":
                            config.update(fallback_lr=config["lr"], fallback_beta2=config["adam_beta2"],
                                          fallback_weight_decay=config["weight_decay"])
                        else:
                            config.update(fallback_lr=baseline["lr"], fallback_beta2=baseline["adam_beta2"],
                                          fallback_weight_decay=baseline["weight_decay"])
                        values.append(config)
    return values


def make_calibration_plan(spec, stage, configs, seeds, tokens, data, sources):
    trials = []
    manifest = verify_data(data, checksums=False)
    for original in configs:
        for seed in seeds:
            config = dict(original, phase="tune", seed=seed, data_seed=seed,
                          total_tokens=tokens, evaluate_test=False)
            step_tokens = Config(**config).tokens_per_step
            config["warmup_tokens"] = max(2, round(tokens * spec["warmup_fraction"] / step_tokens)) * step_tokens
            Config(**config).validate()
            if tokens >= manifest["splits"]["train"]["tokens"]:
                raise ValueError("Calibration exceeds the prepared corpus; do not repeat data silently")
            if config["eval_tokens"] >= manifest["splits"]["val"]["tokens"]:
                raise ValueError("Validation split is too short")
            suffix = fingerprint(dict(config=config, data_id=manifest["data_id"], sources=sources))[:12]
            trials.append(dict(id=f"{stage}-{config['architecture']}-{config['size']}-{config['optimizer']}-s{seed}-{suffix}",
                               candidate=candidate_key(config), config=config))
    random.Random(spec["plan_seed"]).shuffle(trials)
    plan = dict(version=1, phase="calibration", stage=stage, data=str(Path(data).resolve()),
                data_id=manifest["data_id"], sources=sources, trials=trials, selection=None)
    plan["plan_id"] = fingerprint(plan)
    check_plan(plan)
    return plan


def ranked(plans, results):
    """Only terminal validation on every paired seed can select a recipe."""
    groups, platforms, identities = {}, set(), set()
    for plan in plans:
        check_plan(plan)
        identities.add(fingerprint(dict(data_id=plan["data_id"], sources=plan["sources"])))
        for trial in plan["trials"]:
            config = trial["config"]
            directory = Path(results) / trial["id"]
            status = load_json(directory / "status.json")
            loss = None
            if status["status"] == "failed":
                error = status.get("error", "")
                if "non-finite" not in error.lower():
                    raise RuntimeError(f"Infrastructure/unknown failure: {directory}: {error}")
            elif status["status"] == "complete":
                if load_json(directory / "config.json") != config or status["tokens"] != config["total_tokens"]:
                    raise ValueError("Completed config or token budget differs from plan")
                metadata = load_json(directory / "metadata.json")
                expected = dict(config_id=fingerprint(config), data_id=plan["data_id"], sources=plan["sources"])
                if metadata["identity"] != expected:
                    raise ValueError("Completed data/code identity differs")
                platforms.add(fingerprint({k: metadata["hardware"].get(k) for k in
                                           ("device", "name", "capability", "torch", "cuda", "versions")}))
                rows = rows_from(directory / "metrics.jsonl")
                terminal = [r for r in rows if r["kind"] == "val" and r["tokens"] == config["total_tokens"]]
                if not terminal or not math.isfinite(terminal[-1]["loss"]):
                    raise ValueError("No finite terminal validation loss")
                loss = terminal[-1]["loss"]
            else:
                raise RuntimeError(f"Calibration is still {status['status']}: {directory}")
            key = arm_key(config["architecture"], config["size"], config["optimizer"])
            group = groups.setdefault(key, {}).setdefault(trial["candidate"], [])
            group.append(dict(seed=config["seed"], loss=loss, config=config, trial=trial["id"]))
    if len(platforms) != 1 or len(identities) != 1:
        raise ValueError("Calibration mixed hardware/software, source, or data identities")
    ranking = {}
    for arm, candidates_ in groups.items():
        seed_sets = [{r["seed"] for r in runs} for runs in candidates_.values()]
        if any(s != seed_sets[0] for s in seed_sets):
            raise ValueError("Candidates have unequal paired seeds")
        summaries = []
        for key, runs in candidates_.items():
            if len(runs) != len(seed_sets[0]):
                raise ValueError("Duplicate seed for a candidate")
            if len({r["config"]["total_tokens"] for r in runs}) != 1:
                raise ValueError("Candidate used unequal token budgets")
            if any(r["loss"] is None for r in runs):
                continue
            losses = [r["loss"] for r in runs]
            summaries.append(dict(candidate=key, mean_loss=statistics.mean(losses),
                                  sd_loss=statistics.stdev(losses) if len(losses) > 1 else None,
                                  losses=losses, seeds=[r["seed"] for r in runs],
                                  config=runs[0]["config"], trials=[r["trial"] for r in runs]))
        if not summaries:
            raise RuntimeError(f"No stable candidate for {arm}")
        ranking[arm] = sorted(summaries, key=lambda r: (r["mean_loss"], r["candidate"]))
    return ranking, next(iter(platforms))


def boundary_arms(ranking, configs):
    boundaries = []
    for arm, entries in ranking.items():
        winner = entries[0]["config"]
        # Match recipe, including fallback settings, then examine its LR grid.
        def recipe(c):
            c = dict(c, lr=1.0)
            if c["optimizer"] == "adamw":
                c["fallback_lr"] = 1.0
            return candidate_key(c)
        rates = [c["lr"] for c in configs if recipe(c) == recipe(winner)]
        if winner["lr"] in (min(rates), max(rates)):
            boundaries.append(arm)
    return boundaries


def prune_checkpoints(plan, results, keep):
    """Retain metrics/configs for all trials; only complete losers lose weights."""
    for trial in plan["trials"]:
        directory = Path(results) / trial["id"]
        if trial["candidate"] in keep or load_json(directory / "status.json")["status"] != "complete":
            continue
        removed = []
        for path in directory.glob("step_*.pt"):
            removed.append(path.name)
            path.unlink()
        if removed:
            atomic_json(directory / "checkpoint-retention.json", dict(
                reason="Completed calibration candidate not selected; metrics and provenance retained", removed=removed))


def finish_report(spec, root, ranking, horizon, results, unresolved):
    rows, targets, forecast = [], {}, {}
    for trial in horizon["trials"]:
        config = trial["config"]
        key = arm_key(config["architecture"], config["size"], config["optimizer"])
        directory = results / trial["id"]
        status = load_json(directory / "status.json")
        vals = [r for r in rows_from(directory / "metrics.jsonl") if r["kind"] == "val"]
        halfway = min(vals, key=lambda r: abs(r["tokens"] - config["total_tokens"] / 2))
        end = vals[-1]
        rows.append(dict(arm=key, half_loss=halfway["loss"], end_loss=end["loss"],
                         second_half_improvement=halfway["loss"] - end["loss"],
                         train_hours=status["train_seconds"] / 3600,
                         active_hours=status["wall_seconds"] / 3600,
                         confirm_seed_sd=ranking[key][0]["sd_loss"]))
    write_csv(root / "horizon-summary.csv", rows)
    for architecture in spec["architectures"]:
        for size in spec["sizes"]:
            subset = [r for r in rows if r["arm"].startswith(f"{architecture}/{size}/")]
            # Descriptive pilot targets shared by all arms; no arbitrary 2.5 requirement.
            targets[f"{architecture}/{size}"] = math.ceil(max(r["end_loss"] for r in subset) * 10) / 10
    for tokens in (268435456, 536870912, 1073741824):
        forecast[str(tokens)] = dict(
            gpu_hours=sum(r["active_hours"] for r in rows) * tokens / spec["horizon_tokens"] * len(spec["final_seeds"]),
            assumptions="Linear projection from active trainer time; excludes queueing and downtime",
            requires_new_data=tokens > 536870912)
    # Choose among budgets supported by this immutable corpus, using a rule fixed
    # before seeing the longer curves. This does not claim compute-optimal training.
    improvement = statistics.median(r["second_half_improvement"] for r in rows)
    budget = 536870912 if improvement >= 0.05 else 268435456
    configs = [v[0]["config"] for v in ranking.values()]
    final = make_calibration_plan(spec, "final-proposed", configs, spec["final_seeds"],
                                  budget, horizon["data"], horizon["sources"])
    final.pop("plan_id")
    final["phase"] = "final"
    for trial in final["trials"]:
        trial["config"].update(phase="final", evaluate_test=True)
        Config(**trial["config"]).validate()
        trial["id"] += "-" + fingerprint(trial["config"])[:8]
    final["plan_id"] = fingerprint(final)
    check_plan(final)
    if not unresolved:
        atomic_json(root / "final-plan.json", final)
    atomic_json(root / "main-study-readiness.json", dict(
        calibration_complete=True, main_ready=not unresolved, automatic_final_launch=False,
        unresolved=unresolved, suggested_shared_loss_targets=targets,
        proposed_final_tokens=budget, median_second_half_improvement=improvement,
        compute_forecasts=forecast, final_seeds=spec["final_seeds"],
        limitations=["Finite recipe grid, not globally optimal hyperparameters.",
                     "Fallback AdamW is a common control, not a proven best fallback.",
                     "256 Mi-token horizon checks use one fresh seed; the final comparison uses three.",
                     "The selected budget defines a limited-compute study, not converged or compute-optimal models.",
                     "Budgets beyond 512 Mi tokens require new data preparation and calibration."],
        selected_configs={k: v[0]["config"] for k, v in ranking.items()}))


def calibrate(spec_path, root, gpus):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    results = root / "runs"
    spec = load_json(spec_path)
    seeds = [spec[k] for k in ("screen_seeds", "confirm_seeds", "horizon_seeds", "final_seeds")]
    flat = [seed for stage_seeds in seeds for seed in stage_seeds]
    if any(not s for s in seeds) or len(flat) != len(set(flat)) or len(seeds[1]) < 2 or len(seeds[3]) < 3:
        raise ValueError("Use disjoint seeds across phases, at least two confirmation and three final seeds")
    if set(spec["recipes"]) != {"adamw", "muon", "shampoo", "kfac"}:
        raise ValueError("Calibration requires the four agreed optimizer arms")
    for architecture in spec["architectures"]:
        counts = {(len(spec["recipes"][name]), len(rate_for(spec["learning_rates"], name, architecture)))
                  for name in spec["recipes"]}
        if len(counts) != 1:
            raise ValueError("Every optimizer must receive the same recipe and learning-rate search budgets")
    if not 0 < spec["max_hours"] <= 120:
        raise ValueError("Calibration wall-time cap must be positive and at most 120 hours")
    sources = source_identity()
    manifest = verify_data(spec["data"])
    identity = dict(spec=spec, sources=sources, data_id=manifest["data_id"],
                    runner_sources={n: sha256_file(Path(__file__).with_name(n)) for n in
                                    ("calibration.py", "study.py", "report.py")})
    path = root / "campaign.json"
    if path.exists():
        campaign = load_json(path)
        if campaign["identity"] != identity:
            raise ValueError("Campaign source/spec/data changed; use a new campaign directory")
    else:
        now = datetime.now(timezone.utc)
        deadline = min(now.timestamp() + spec["max_hours"] * 3600,
                       datetime.fromisoformat(spec["stop_before"]).timestamp())
        campaign = dict(identity=identity, started_at=now.isoformat(), deadline=deadline)
        atomic_json(path, campaign)
    deadline = campaign["deadline"]
    remaining = lambda: max(0, (deadline - time.time()) / 3600)
    platform = None
    unresolved = []

    def stage(name, configs, seeds, tokens):
        nonlocal platform
        plan = make_calibration_plan(spec, name, configs, seeds, tokens, spec["data"], sources)
        path = root / f"{name}.json"
        if path.exists():
            if load_json(path) != plan:
                raise ValueError(f"Frozen stage differs: {name}")
        else:
            atomic_json(path, plan)
        atomic_json(root / "status.json", dict(status="running", stage=name, deadline=deadline))
        print(f"CALIBRATION {name}: {len(plan['trials'])} trials; {remaining():.1f} hours remaining", flush=True)
        if remaining() <= 0:
            raise TimeoutError("Calibration deadline reached; weights and completed results retained")
        try:
            complete = run_plan(path, results, gpus, hours=remaining())
        except RuntimeError:
            # Only observed numerical divergence may be scored as an unsuccessful candidate.
            # ranked rejects infrastructure failures and all unfinished trials.
            complete = True
            ranked([plan], results)
        if not complete:
            raise TimeoutError("Calibration paused; rerun the same command to resume before the deadline")
        ranking, current_platform = ranked([plan], results)
        if platform is not None and current_platform != platform:
            raise ValueError("Hardware/software stack changed between calibration stages")
        platform = current_platform
        atomic_json(root / f"{name}-ranking.json", ranking)
        return plan, ranking

    selected = {}
    with file_lock(root / ".campaign.lock"):
        for label, names in (("adamw", ["adamw"]), ("others", ["muon", "shampoo", "kfac"])):
            configs = candidates(spec, names, selected or None)
            plan, ranking = stage(label + "-screen", configs, spec["screen_seeds"], spec["screen_tokens"])
            screen_plans = [plan]
            # Every arm gets the same one-time wider bracket, even if its initial
            # winner was interior. No optimizer receives extra free search trials.
            extra = candidates(spec, names, selected or None, expansion=True)
            plan, _ = stage(label + "-expand", extra, spec["screen_seeds"], spec["screen_tokens"])
            screen_plans.append(plan)
            configs += extra
            ranking, _ = ranked(screen_plans, results)
            shortlist = [entry["config"] for entries in ranking.values() for entry in entries[:2]]
            if any(len(entries) < 2 for entries in ranking.values()):
                raise RuntimeError("Fewer than two stable candidates; calibration needs a revised search")
            for plan in screen_plans:
                prune_checkpoints(plan, results, {candidate_key(c) for c in shortlist})
            confirm, confirmed = stage(label + "-confirm", shortlist, spec["confirm_seeds"], spec["confirm_tokens"])
            for arm in boundary_arms(confirmed, configs):
                unresolved.append(f"Confirmed LR is still at the searched boundary: {arm}")
            selected.update(confirmed)
            prune_checkpoints(confirm, results, {v[0]["candidate"] for v in confirmed.values()})
            atomic_json(root / "selected.json", selected)
        horizon, _ = stage("horizon", [v[0]["config"] for v in selected.values()],
                           spec["horizon_seeds"], spec["horizon_tokens"])
        finish_report(spec, root, selected, horizon, results, unresolved)
        atomic_json(root / "status.json", dict(status="complete", main_study_started=False))
        print("Calibration complete. Review main-study-readiness.json; final training was not launched.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="calibration.json")
    parser.add_argument("--output", default="results/calibration")
    parser.add_argument("--gpus", nargs="+", required=True)
    args = parser.parse_args()
    try:
        calibrate(args.spec, args.output, args.gpus)
    except Exception as exc:
        root = Path(args.output)
        if root.exists():
            status = load_json(root / "status.json") if (root / "status.json").exists() else {}
            atomic_json(root / "status.json", status | dict(
                status="paused" if isinstance(exc, TimeoutError) else "failed", error=str(exc)))
        raise


if __name__ == "__main__":
    main()
