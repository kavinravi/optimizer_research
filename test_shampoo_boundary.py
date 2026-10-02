"""CPU-only recorded-output check: python test_shampoo_boundary.py."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch

from artifacts import atomic_json, fingerprint, sha256_file
from calibration import make_calibration_plan, ranked
from shampoo_boundary import ARMS, follow_up
from study import arm_key, check_plan, load_json
from train import Config


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base = root / "calibration"
        runs = base / "runs"
        runs.mkdir(parents=True)
        spec = load_json("calibration.json") | dict(data=str(root / "data"), screen_tokens=64,
                                                   confirm_tokens=128, horizon_tokens=256)
        sources = {"training": "fixture"}
        manifest = dict(data_id="fixture", splits=dict(train=dict(tokens=536870913), val=dict(tokens=1048577)))

        def key(c):
            return arm_key(c["architecture"], c["size"], c["optimizer"])

        def configs(names, rates, confirmation=False):
            result = []
            for architecture in spec["architectures"]:
                for size in spec["sizes"]:
                    for optimizer in names:
                        arm = arm_key(architecture, size, optimizer)
                        used = [0.001 / 3, 0.001] if confirmation and arm in ARMS else rates
                        for lr in used:
                            c = Config(architecture=architecture, size=size, optimizer=optimizer,
                                       lr=lr, fallback_lr=lr if optimizer == "adamw" else .001,
                                       sequence=8, accumulation=1, eval_tokens=64)
                            result.append(asdict(c))
            return result

        def record(plan, loss_for):
            check_plan(plan)
            for t in plan["trials"]:
                d = runs / t["id"]
                if (d / "status.json").exists():
                    continue
                d.mkdir()
                c = t["config"]
                loss = loss_for(c)
                atomic_json(d / "config.json", c)
                atomic_json(d / "status.json", dict(status="complete", tokens=c["total_tokens"],
                                                     train_seconds=2, wall_seconds=3))
                atomic_json(d / "metadata.json", dict(identity=dict(config_id=fingerprint(c),
                            data_id=plan["data_id"], sources=sources), hardware=dict(device="cpu")))
                (d / "metrics.jsonl").write_text("\n".join(json.dumps(dict(kind="val", tokens=n, loss=v))
                    for n, v in [(c["total_tokens"] // 2, loss + .1), (c["total_tokens"], loss)]) + "\n")

        with patch("calibration.verify_data", return_value=manifest), \
             patch("shampoo_boundary.source_identity", return_value=sources):
            confirms = []
            for label, names in [("adamw", ["adamw"]), ("others", ["muon", "shampoo", "kfac"])]:
                for stage, rates in [("screen", [.001, .003]), ("expand", [.001 / 3, .009]),
                                     ("confirm", [.001, .003])]:
                    confirmation = stage == "confirm"
                    p = make_calibration_plan(spec, label + "-" + stage, configs(names, rates, confirmation),
                        spec["confirm_seeds"] if confirmation else spec["screen_seeds"],
                        spec["confirm_tokens"] if confirmation else spec["screen_tokens"], spec["data"], sources)
                    atomic_json(base / f"{label}-{stage}.json", p)
                    if confirmation:
                        record(p, lambda c: 3.0 if c["lr"] == (.001 / 3 if key(c) in ARMS else .001) else 4.0)
                        confirms.append(p)
            selected, _ = ranked(confirms, runs)
            atomic_json(base / "selected.json", selected)
            horizon = make_calibration_plan(spec, "horizon", [e[0]["config"] for e in selected.values()],
                                            spec["horizon_seeds"], spec["horizon_tokens"], spec["data"], sources)
            atomic_json(base / "horizon.json", horizon)
            record(horizon, lambda c: 3.0)
            atomic_json(base / "campaign.json", dict(deadline=time.time() + 3600, identity=dict(
                spec=spec, sources=sources, runner_sources={n: sha256_file(n) for n in
                                                          ("calibration.py", "study.py", "report.py")})))
            atomic_json(base / "status.json", dict(status="complete"))
            atomic_json(base / "main-study-readiness.json", dict(unresolved=[
                f"Confirmed LR is still at the searched boundary: {arm}" for arm in ARMS]))
            originals = {p: p.read_bytes() for p in base.glob("*.json")}
            original_run_dirs = {p.name for p in runs.iterdir()}

            for outcome in ("keep", "replace", "still-boundary"):
                output = root / outcome
                calls = []

                def execute(path, results, gpus, hours):
                    assert 0 < hours <= 1 and Path(results) == runs and gpus == ["cpu"]
                    p = load_json(path)
                    calls.append(p)
                    for t in p["trials"]:
                        c = t["config"]
                        assert key(c) in ARMS and not c["evaluate_test"] and c["phase"] == "tune"
                        original = selected[key(c)][0]["config"]
                        assert all(c[k] == original[k] for k in c if k not in (
                            "lr", "seed", "data_seed", "total_tokens", "warmup_tokens"))
                    record(p, lambda c: 5.0 if outcome == "keep" else
                           2.0 if c["lr"] == (.001 / 3) / 3 else
                           1.0 if outcome == "still-boundary" else 4.0)
                    return True

                with patch("shampoo_boundary.run_plan", side_effect=execute):
                    follow_up(base, output, ["cpu"], prepare_only=True)
                    assert not calls
                    follow_up(base, output, ["cpu"])
                assert len(calls[0]["trials"]) == 8
                assert {t["config"]["seed"] for t in calls[0]["trials"]} == {1337, 7331}
                status = load_json(output / "status.json")
                assert not status["main_study_started"]
                if outcome == "still-boundary":
                    assert status["status"] == "needs-review" and set(status["unresolved"]) == set(ARMS)
                    assert len(calls) == 1 and not (output / "final-plan.json").exists()
                else:
                    assert status["status"] == "complete"
                    assert load_json(output / "main-study-readiness.json")["main_ready"]
                    combined = load_json(output / "combined-horizon.json")
                    check_plan(combined)
                    assert len(combined["trials"]) == 16
                    assert len(calls) == (2 if outcome == "replace" else 1)
                    if outcome == "replace":
                        assert len(calls[1]["trials"]) == 2
                        assert {t["config"]["seed"] for t in calls[1]["trials"]} == {9001}
                    else:
                        assert combined == horizon
                        with patch("shampoo_boundary.run_plan", return_value=False):
                            try:
                                follow_up(base, root / "paused", ["cpu"])
                                raise AssertionError("Paused queue advanced to selection")
                            except TimeoutError:
                                pass
                        with patch("shampoo_boundary.run_plan", side_effect=execute):
                            follow_up(base, root / "paused", ["cpu"])
                assert all(p.read_bytes() == content for p, content in originals.items())
                # Isolate the recorded outcomes while retaining original campaign artifacts.
                for d in runs.iterdir():
                    if d.name not in original_run_dirs:
                        shutil.rmtree(d)

            with patch("shampoo_boundary.time.time", return_value=time.time() + 7200), \
                 patch("shampoo_boundary.run_plan") as runner:
                try:
                    follow_up(base, root / "expired", ["cpu"])
                    raise AssertionError("Expired deadline was ignored")
                except TimeoutError:
                    pass
                runner.assert_not_called()
    print("Shampoo follow-up checks passed: paired selection, conditional horizons, boundary/deadline gates and resume.")


if __name__ == "__main__":
    main()
