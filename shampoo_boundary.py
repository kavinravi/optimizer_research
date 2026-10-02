"""Finish the two Shampoo LR boundary checks; never launch the main study."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from artifacts import atomic_json, file_lock, fingerprint, sha256_file
from calibration import boundary_arms, finish_report, make_calibration_plan, ranked
from study import arm_key, check_plan, load_json, run_plan
from train import source_identity


ARMS = ("transformer/150m/shampoo", "mamba/300m/shampoo")


def freeze(path, value):
    if path.exists():
        if load_json(path) != value:
            raise ValueError(f"Frozen follow-up differs: {path}")
    else:
        atomic_json(path, value)


def follow_up(base, output, gpus, prepare_only=False):
    base, output = Path(base), Path(output)
    if base.resolve() == output.resolve():
        raise ValueError("Write the amendment separately from the original campaign")
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".campaign.lock"):
        campaign = load_json(base / "campaign.json")
        spec = campaign["identity"]["spec"]
        sources = source_identity()
        if campaign["identity"]["sources"] != sources:
            raise ValueError("Training/optimizer code differs from the original campaign")
        if any(sha256_file(name) != digest for name, digest in
               campaign["identity"]["runner_sources"].items()):
            raise ValueError("Original calibration/runner code changed")
        if load_json(base / "status.json")["status"] != "complete":
            raise ValueError("The original calibration must finish first")
        readiness = load_json(base / "main-study-readiness.json")
        expected = {f"Confirmed LR is still at the searched boundary: {arm}" for arm in ARMS}
        if set(readiness["unresolved"]) != expected:
            raise ValueError("This follow-up only resolves the two recorded Shampoo boundaries")
        deadline = campaign["deadline"]
        baseline = load_json(base / "selected.json")
        results = base / "runs"
        confirms = [load_json(base / f"{label}-confirm.json") for label in ("adamw", "others")]
        original, platform = ranked(confirms, results)
        if original != baseline:
            raise ValueError("Original selections do not match the recorded paired confirmations")
        screened = [t["config"] for label in ("adamw", "others") for stage in ("screen", "expand")
                    for t in load_json(base / f"{label}-{stage}.json")["trials"]]
        configs = [dict(baseline[arm][0]["config"], lr=baseline[arm][0]["config"]["lr"] / divisor)
                   for arm in ARMS for divisor in (3, 9)]
        source_files = ["campaign.json", "selected.json", "main-study-readiness.json", "horizon.json"]
        source_files += [f"{label}-{stage}.json" for label in ("adamw", "others")
                         for stage in ("screen", "expand", "confirm")]
        freeze(output / "amendment.json", dict(
            base=str(base.resolve()), base_files={n: sha256_file(base / n) for n in source_files},
            controller_sha256=sha256_file(__file__), deadline=deadline,
            paired_seeds=spec["confirm_seeds"], tokens=spec["confirm_tokens"],
            learning_rates={arm: [baseline[arm][0]["config"]["lr"] / d for d in (3, 9)] for arm in ARMS},
            rationale="Extend only the two unresolved lower LR boundaries; reuse paired controls.",
            fairness="Two extra paired LR candidates per affected arm; report this unequal extra tuning cost.",
            selection="Mean terminal confirmation loss; no selection using horizon/test losses.",
            automatic_final_launch=False))
        plan = make_calibration_plan(spec, "shampoo-boundary-confirm", configs,
                                     spec["confirm_seeds"], spec["confirm_tokens"], spec["data"], sources)
        freeze(output / "confirm.json", plan)
        print(f"{len(plan['trials'])} paired confirmation runs; deadline "
              f"{datetime.fromtimestamp(deadline, timezone.utc).isoformat()}", flush=True)
        if prepare_only:
            return

        def stage(name, plan):
            path = output / f"{name}.json"
            freeze(path, plan)
            atomic_json(output / "status.json", dict(status="running", stage=name, deadline=deadline,
                                                      main_study_started=False))
            remaining = (deadline - time.time()) / 3600
            if remaining <= 0:
                raise TimeoutError("Maintenance cutoff reached; rerun after explicitly scheduling continuation")
            try:
                complete = run_plan(path, results, gpus, hours=remaining)
            except RuntimeError:
                # Only terminal numerical failures may be scored as unsuccessful candidates.
                ranked([plan], results)
                complete = True
            if not complete:
                raise TimeoutError("Follow-up paused with checkpoints; rerun the same command to resume")
            if ranked([plan], results)[1] != platform:
                raise ValueError("Follow-up hardware/software differs from the original campaign")

        stage("confirm", plan)
        selected, _ = ranked(confirms + [plan], results)
        atomic_json(output / "selected.json", selected)
        unresolved = boundary_arms(selected, screened + configs)
        review = dict(
            automatic_final_launch=False, unresolved=unresolved,
            extra_confirm_runs=len(plan["trials"]),
            extra_confirm_gpu_hours=sum(load_json(results / t["id"] / "status.json").get("wall_seconds", 0)
                                        for t in plan["trials"]) / 3600,
            comparisons={arm: dict(old_lr=baseline[arm][0]["config"]["lr"],
                                   old_mean=baseline[arm][0]["mean_loss"],
                                   new_lr=selected[arm][0]["config"]["lr"],
                                   new_mean=selected[arm][0]["mean_loss"],
                                   candidates=[dict(lr=e["config"]["lr"], mean=e["mean_loss"],
                                                    sd=e["sd_loss"], seeds=e["seeds"], trials=e["trials"])
                                               for e in selected[arm]]) for arm in ARMS})
        atomic_json(output / "boundary-review.json", review)
        if unresolved:
            atomic_json(output / "status.json", dict(status="needs-review", stage="confirm",
                                                      unresolved=unresolved, main_study_started=False))
            print(f"Paired checks finished; boundaries remain: {unresolved}. Stop for review.", flush=True)
            return
        changed = [arm for arm in ARMS if selected[arm][0]["candidate"] != baseline[arm][0]["candidate"]]
        horizon = load_json(base / "horizon.json")
        if changed:
            replacement = make_calibration_plan(spec, "shampoo-boundary-horizon",
                                                [selected[arm][0]["config"] for arm in changed],
                                                spec["horizon_seeds"], spec["horizon_tokens"], spec["data"], sources)
            stage("replacement-horizon", replacement)
            horizon = dict(horizon, stage="horizon-after-shampoo-boundary", trials=[
                t for t in horizon["trials"] if arm_key(t["config"]["architecture"],
                t["config"]["size"], t["config"]["optimizer"]) not in changed] + replacement["trials"])
            horizon.pop("plan_id")
            horizon["plan_id"] = fingerprint(horizon)
        check_plan(horizon)
        freeze(output / "combined-horizon.json", horizon)
        _, current_platform = ranked([horizon], results)
        if current_platform != platform:
            raise ValueError("Combined horizon hardware/software differs")
        finish_report(spec, output, selected, horizon, results, [])
        atomic_json(output / "status.json", dict(status="complete", main_study_started=False,
                                                  changed_arms=changed, extra_confirm_runs=len(plan["trials"])))
        print("Shampoo follow-up complete. Review this directory's readiness report; main study not launched.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="results/calibration")
    parser.add_argument("--output", default="results/shampoo-boundary")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        follow_up(args.base, args.output, args.gpus, args.prepare_only)
    except Exception as exc:
        root = Path(args.output)
        if root.exists():
            previous = load_json(root / "status.json") if (root / "status.json").exists() else {}
            atomic_json(root / "status.json", previous | dict(
                status="paused" if isinstance(exc, TimeoutError) else "failed", error=str(exc),
                main_study_started=False))
        raise


if __name__ == "__main__":
    main()
