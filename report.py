"""Export validation curves and repeated-seed summaries without mixing platforms."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from artifacts import atomic_json, fingerprint


def rows_from(path):
    """A live writer may have an incomplete final line; other corruption is an error."""
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b"\n")]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def collect(results, target=None):
    curves, runs, comparisons = [], [], {}
    for path in sorted(Path(results).glob("*/config.json")):
        directory = path.parent
        if not (directory / "metadata.json").exists():
            continue
        config = json.loads(path.read_text())
        metadata = json.loads((directory / "metadata.json").read_text())
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "starting"}
        hardware = {k: metadata["hardware"].get(k) for k in
                    ("device", "name", "capability", "torch", "cuda", "versions")}
        protocol = {k: v for k, v in config.items() if k not in
                    ("optimizer", "lr", "fallback_lr", "seed", "data_seed", "log_every", "checkpoint_every")}
        # Checkpoint cadence affects elapsed time, so retain it in comparison identity.
        protocol["checkpoint_every"] = config["checkpoint_every"]
        comparison = dict(protocol=protocol, hardware=hardware, data_id=metadata["identity"]["data_id"],
                          sources=metadata["identity"]["sources"])
        group = fingerprint(comparison)[:12]
        comparisons[group] = comparison
        arm = f"{config['optimizer']}-lr{config['lr']:g}-fallback{config['fallback_lr']:g}"
        common = dict(comparison=group, trial=directory.name, architecture=config["architecture"],
                      size=config["size"], optimizer=config["optimizer"], phase=config["phase"],
                      seed=config["seed"], lr=config["lr"], fallback_lr=config["fallback_lr"],
                      hardware=hardware.get("name") or hardware["device"], precision=config["precision"], arm=arm)
        metrics = rows_from(directory / "metrics.jsonl")
        validations = [r for r in metrics if r["kind"] == "val"]
        for row in validations:
            curves.append(dict(common, tokens=row["tokens"], loss=row["loss"],
                               train_seconds=row["train_seconds"], wall_seconds=row["wall_seconds"],
                               status=status["status"]))
        latest = metrics[-1] if metrics else {}
        terminal = next((r for r in reversed(validations) if r["tokens"] == config["total_tokens"]), None)
        hit = next((r for r in validations if target is not None and r["loss"] <= target), None)
        attempts = rows_from(directory / "attempts.jsonl")
        runs.append(dict(common, status=status["status"], tokens=latest.get("tokens", 0),
                         planned_tokens=config["total_tokens"],
                         terminal_val_loss=terminal["loss"] if terminal else None,
                         train_seconds=latest.get("train_seconds"), wall_seconds=latest.get("wall_seconds"),
                         recorded_attempt_seconds=sum(r["seconds"] for r in attempts) if attempts else None,
                         test_loss=status.get("test_loss"), target=target,
                         first_target_tokens=hit["tokens"] if hit else None,
                         first_target_train_seconds=hit["train_seconds"] if hit else None,
                         peak_allocated_bytes=max((r.get("peak_allocated_bytes") or 0 for r in metrics), default=0)))
    summaries = []
    for group, arm in sorted({(r["comparison"], r["arm"]) for r in runs}):
        selected = [r for r in runs if r["comparison"] == group and r["arm"] == arm]
        if len({r["seed"] for r in selected}) != len(selected):
            raise ValueError(f"Duplicate seeds in {group}/{arm}; report separate experiments separately")
        complete = [r for r in selected if r["status"] == "complete" and r["terminal_val_loss"] is not None]
        losses = [r["terminal_val_loss"] for r in complete]
        summaries.append(dict(comparison=group, arm=arm, runs=len(selected), complete=len(complete),
                              failed=sum(r["status"] == "failed" for r in selected),
                              complete_seeds=",".join(str(r["seed"]) for r in complete),
                              mean_terminal_val_loss=statistics.mean(losses) if losses else None,
                              sd_terminal_val_loss=statistics.stdev(losses) if len(losses) > 1 else None,
                              mean_train_seconds=statistics.mean(r["train_seconds"] for r in complete) if complete else None))
    return curves, runs, summaries, comparisons


def report(results, output, target=None, plots=True):
    if target is not None and not math.isfinite(target):
        raise ValueError("Target must be finite")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    curves, runs, summaries, comparisons = collect(results, target)
    if not runs:
        raise ValueError("No training runs found")
    write_csv(output / "validation.csv", curves)
    write_csv(output / "runs.csv", runs)
    write_csv(output / "summary.csv", summaries)
    atomic_json(output / "comparisons.json", comparisons)
    if plots:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("XDG_CACHE_HOME", tempfile.gettempdir())) /
                                                 f"matplotlib-{os.getuid()}"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for group, info in comparisons.items():
            figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            group_rows = [r for r in curves if r["comparison"] == group]
            arms = sorted({r["arm"] for r in group_rows})
            for arm in arms:
                rows = [r for r in group_rows if r["arm"] == arm]
                color = f"C{arms.index(arm) % 10}"
                for trial in sorted({r["trial"] for r in rows}):
                    points = sorted((r for r in rows if r["trial"] == trial), key=lambda r: r["tokens"])
                    for axis, field, scale in zip(axes, ("tokens", "train_seconds", "wall_seconds"), (1e6, 3600, 3600)):
                        axis.plot([p[field] / scale for p in points], [p["loss"] for p in points],
                                  color=color, alpha=.65, linewidth=1,
                                  label=f"{arm} s{points[0]['seed']}")
            for axis, label in zip(axes, ("Training tokens (millions)", "Training hours", "Active elapsed hours")):
                axis.set(xlabel=label, ylabel="Validation cross-entropy (nats/token)")
                axis.grid(alpha=.2)
                if target is not None:
                    axis.axhline(target, color="black", linestyle=":", linewidth=1)
            if arms:
                axes[-1].legend(fontsize=6)
            p = info["protocol"]
            figure.suptitle(f"{p['architecture']} {p['size']} {p['phase']} | "
                           f"{info['hardware'].get('name') or info['hardware']['device']} | {group}")
            figure.savefig(output / f"{group}.png", dpi=160)
            figure.savefig(output / f"{group}.pdf")
            plt.close(figure)
    for row in runs:
        print(f"{row['status']:8} {row['tokens']:>13,}/{row['planned_tokens']:,}  {row['trial']}")
    print(f"Report: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/training")
    parser.add_argument("--output", default="results/report")
    parser.add_argument("--target", type=float)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    report(args.results, args.output, args.target, not args.no_plots)
