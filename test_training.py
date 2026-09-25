"""Run with python test_training.py; add --gpu for Mamba/CUDA resume checks."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from artifacts import atomic_json, fingerprint, sha256_file
import prepare_data
from prepare_data import document_split, prepare, verify_data
from train import Config, evaluate, model_for, read_checkpoint, rng_state, train
from optimizers import optimizer_for
from study import check_plan, make_plan, select, run_plan
from report import collect, report

GPU = "--gpu" in sys.argv
if GPU:
    sys.argv.remove("--gpu")


def fixture(root):
    tokenizer = Tokenizer(models.WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
                                          "alpha": 4, "beta": 5, "gamma": 6, "delta": 7,
                                          "document": 8}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    texts = [("alpha beta gamma delta " * 20) + f"document {i}" for i in range(2000)]
    path = root / "fixture.parquet"
    pq.write_table(pa.table({"text": texts}), path)
    recipe = dict(dataset="local-fixture", revision=None, files=[str(path)],
                  tokenizer_sha256=sha256_file(tokenizer_path),
                  budgets=dict(train=4097, val=513, test=513))
    return tokenizer_path, recipe


def assert_nested(test, a, b, *, approximate=False):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=1e-4 if approximate else 0, atol=1e-5 if approximate else 0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_nested(test, a[key], b[key], approximate=approximate)
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_nested(test, x, y, approximate=approximate)
    elif isinstance(a, float) and approximate:
        test.assertAlmostEqual(a, b, delta=max(1e-5, abs(a) * 1e-4))
    else:
        test.assertEqual(a, b)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.tokenizer, cls.recipe = fixture(cls.root)
        cls.data = cls.root / "data"
        prepare(cls.data, cls.tokenizer, cls.recipe)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def config(self, name="adamw", **kwargs):
        return Config(phase="verification", tiny=True, device="cpu", precision="fp32", sequence=8,
                      micro_batch=1, accumulation=2, total_tokens=96, warmup_tokens=32,
                      eval_tokens=64, eval_every=2, checkpoint_every=2, log_every=100, refresh=2,
                      optimizer=name, lr={"adamw": 1e-3, "muon": .01, "shampoo": .003, "kfac": .0003}[name],
                      fallback_lr=1e-3, **kwargs)

    def test_data_resume_and_integrity(self):
        output = self.root / "interrupted-data"
        original = prepare_data.source_batches

        def interrupted(*args):
            for batch in original(*args):
                yield batch
                raise RuntimeError("simulated interruption")

        with patch("prepare_data.source_batches", interrupted), self.assertRaisesRegex(RuntimeError, "simulated"):
            prepare(output, self.tokenizer, self.recipe)
        prepare(output, self.tokenizer, self.recipe)
        for split in prepare_data.SPLITS:
            self.assertEqual((output / f"{split}.bin").read_bytes(), (self.data / f"{split}.bin").read_bytes())
        self.assertEqual(document_split("cafe\u0301  text"), document_split("café\ntext"))
        with open(output / "train.bin", "r+b") as stream:
            stream.write(b"\xff\xff")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_data(output)

    def test_exact_cpu_resume_all_optimizers(self):
        for name in ("adamw", "muon", "shampoo", "kfac"):
            with self.subTest(optimizer=name):
                config = self.config(name)
                straight, resumed = self.root / f"{name}-straight", self.root / f"{name}-resumed"
                train(config, self.data, straight)
                status = train(config, self.data, resumed, stop_after_steps=3)
                self.assertEqual(status["status"], "paused")
                train(config, self.data, resumed, resume=True)
                a, _ = read_checkpoint(straight)
                b, _ = read_checkpoint(resumed)
                for key in ("model", "optimizer", "sampler", "scheduler", "rng", "fisher_rng"):
                    assert_nested(self, a[key], b[key])
                self.assertEqual(a["tokens"], config.total_tokens)
                self.assertEqual(a["counters"]["last_eval_step"], 6)
                for path in (straight, resumed):
                    self.assertLessEqual(len(list(path.glob("step_*.pt"))), 2)
                    rows = [json.loads(s) for s in (path / "metrics.jsonl").read_text().splitlines()]
                    self.assertEqual([r["tokens"] for r in rows if r["kind"] == "train"], list(range(16, 97, 16)))
                    self.assertTrue(all(r["train_seconds"] >= 0 and r["wall_seconds"] >= r["train_seconds"] for r in rows))
                with self.assertRaisesRegex(ValueError, "config"):
                    train(replace(config, total_tokens=112), self.data, resumed, resume=True)

    def test_validation_preserves_rng(self):
        config = self.config()
        model = model_for("transformer", "150m", 8, True, vocab_size=9)
        before = rng_state()
        value = evaluate(model, self.data / "val.bin", config, torch.device("cpu"))
        self.assertTrue(np.isfinite(value))
        assert_nested(self, before, rng_state())
        self.assertTrue(model.training)

    def test_identical_routing(self):
        model = model_for("transformer", "150m", 8, True, vocab_size=9)
        expected = None
        for name in ("adamw", "muon", "shampoo", "kfac"):
            optimizer, routing = optimizer_for(model, name, lr=1e-3, fallback_lr=1e-3)
            assignment = [(g["selected"], g["names"], g["weight_decay"]) for g in routing]
            if expected is not None:
                self.assertEqual(sorted(assignment), sorted(expected))
            expected = assignment
            self.assertEqual(sum(g["parameters"] for g in routing), sum(p.numel() for p in model.parameters()))
            if name == "kfac":
                optimizer.tracker.remove()

    def test_config_rejects_invalid_budgets(self):
        for config in (replace(self.config(), total_tokens=97), replace(self.config(), warmup_tokens=96),
                       replace(self.config(), fallback_lr=.002), replace(self.config(), tiny=False, evaluate_test=True)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_lr_selection_uses_terminal_validation_and_all_seeds(self):
        spec = json.loads(Path(__file__).with_name("study.json").read_text())
        spec["training"].update(sequence=8, micro_batch=1, accumulation=2, eval_tokens=64)
        plan = make_plan(spec, "tune-adamw", self.data, architectures=["transformer"], sizes=["150m"], tokens=96)
        check_plan(plan)
        plans = self.root / "plan.json"
        atomic_json(plans, plan)
        results = self.root / "selection-results"
        rates = sorted({t["config"]["lr"] for t in plan["trials"]})
        for trial in plan["trials"]:
            d = results / trial["id"]
            d.mkdir(parents=True)
            config = trial["config"]
            atomic_json(d / "config.json", config)
            atomic_json(d / "metadata.json", dict(identity=dict(data_id=plan["data_id"], sources=plan["sources"]), hardware={}))
            atomic_json(d / "status.json", dict(status="complete", tokens=96))
            loss = 1.0 if config["lr"] == rates[1] else 2.0
            (d / "metrics.jsonl").write_text(json.dumps(dict(kind="test", tokens=96, loss=-100)) + "\n" +
                                             json.dumps(dict(kind="val", tokens=96, loss=loss)) + "\n")
        selected = select([plans], results)
        self.assertEqual(selected["winners"]["transformer/150m/adamw"]["lr"], rates[1])
        atomic_json(results / plan["trials"][0]["id"] / "status.json", dict(status="paused"))
        with self.assertRaisesRegex(ValueError, "paused"):
            select([plans], results)

    def test_queue_resumes_and_report_separates_hardware(self):
        config = self.config()
        directory = self.root / "queue-results"
        trial = directory / "trial-one"
        train(config, self.data, trial, stop_after_steps=2)
        from train import source_identity
        plan = dict(phase="verification", data=str(self.data), data_id=verify_data(self.data)["data_id"],
                    sources=source_identity(), trials=[dict(id=trial.name, config=asdict(config))])
        plan["plan_id"] = fingerprint(plan)
        path = self.root / "queue-plan.json"
        atomic_json(path, plan)
        run_plan(path, directory, ["cpu"])
        self.assertEqual(json.loads((trial / "status.json").read_text())["status"], "complete")
        with patch("study.subprocess.Popen", side_effect=AssertionError("Completed trial must be skipped")):
            run_plan(path, directory, ["cpu"])
        report(directory, self.root / "report", target=100, plots=True)
        curves, runs, summaries, comparisons = collect(directory, 100)
        self.assertEqual(runs[0]["first_target_tokens"], 0)
        self.assertEqual(summaries[0]["complete"], 1)
        self.assertTrue(list((self.root / "report").glob("*.pdf")))
        import shutil
        copy = directory / "other-gpu"
        shutil.copytree(trial, copy)
        metadata = json.loads((copy / "metadata.json").read_text())
        metadata["hardware"]["name"] = "different GPU"
        atomic_json(copy / "metadata.json", metadata)
        self.assertEqual(len(collect(directory)[3]), 2)
        with self.assertRaisesRegex(ValueError, "disagree"):
            run_plan(path, directory, ["cpu", "GPU-00000000-0000-0000-0000-000000000000"])

    def test_uncommitted_metrics_are_removed_on_resume(self):
        config = self.config()
        path = self.root / "discard-test"
        train(config, self.data, path, stop_after_steps=1)
        with open(path / "metrics.jsonl", "a") as log:
            log.write('{"kind":"train","tokens":999999}\n')
        train(config, self.data, path, resume=True)
        self.assertNotIn("999999", (path / "metrics.jsonl").read_text())
        self.assertEqual(len(list(path.glob("discarded-*.jsonl"))), 1)

    def test_final_checkpoint_recovers_without_repeating_test(self):
        config = replace(self.config(), tiny=False, phase="final", evaluate_test=True)
        output = self.root / "final-recovery"
        original_factory = model_for

        def small_factory(arch, size, sequence, tiny, **kwargs):
            return original_factory(arch, size, sequence, True, **kwargs)

        # Exercise the final-run path with the small fixture, then simulate loss
        # of status.json after the final checkpoint was committed.
        with patch("train.DATASET", "local-fixture"), patch("train.model_for", small_factory):
            train(config, self.data, output)
            (output / "status.json").unlink()
            with patch("train.evaluate", side_effect=AssertionError("Final evaluation was already committed")):
                recovered = train(config, self.data, output, resume=True)
        self.assertIsNotNone(recovered["test_loss"])
        rows = [json.loads(s) for s in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(sum(r["kind"] == "test" for r in rows), 1)

    def test_tuning_and_final_plans_freeze_fallback_and_repeat_seeds(self):
        spec = json.loads(Path(__file__).with_name("study.json").read_text())
        spec["training"].update(sequence=8, micro_batch=1, accumulation=2, eval_tokens=64)
        args = dict(architectures=["transformer"], sizes=["150m"], tokens=96)
        baseline = make_plan(spec, "tune-adamw", self.data, **args)
        selection = dict(data_id=baseline["data_id"], sources=baseline["sources"], winners={
            "transformer/150m/adamw": dict(lr=.001, fallback_lr=.001, tokens=96, seeds=[3619, 1337])})
        advanced = make_plan(spec, "tune-others", self.data, selections=selection, **args)
        self.assertEqual(len(advanced["trials"]), 3 * len(baseline["trials"]))
        self.assertEqual({t["config"]["fallback_lr"] for t in advanced["trials"]}, {.001})
        for optimizer in ("muon", "shampoo", "kfac"):
            selection["winners"][f"transformer/150m/{optimizer}"] = dict(
                lr=.003, fallback_lr=.001, tokens=96, seeds=[3619, 1337])
        final = make_plan(spec, "final", self.data, selections=selection, **args)
        self.assertEqual(len(final["trials"]), 12)
        self.assertTrue(all(t["config"]["evaluate_test"] for t in final["trials"]))
        for phase, changes in (("tune-others", dict(tokens=112)), ("final", dict(seeds=[3619, 101])),
                               ("final", dict(tokens=8192))):
            with self.assertRaises(ValueError):
                make_plan(spec, phase, self.data, selections=selection, **(args | changes))
        final["trials"][0]["config"]["total_tokens"] = 112
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            check_plan(final)

    @unittest.skipUnless(GPU, "Pass --gpu for CUDA/Mamba integration checks")
    def test_gpu_resume_all_architectures_and_optimizers(self):
        self.assertTrue(torch.cuda.is_available())
        for architecture in ("transformer", "mamba"):
            for name in ("adamw", "muon", "shampoo", "kfac"):
                with self.subTest(architecture=architecture, optimizer=name):
                    config = replace(self.config(name), architecture=architecture, device="cuda", precision="bf16",
                                     sequence=32, total_tokens=256, warmup_tokens=64, eval_tokens=64)
                    tag = f"gpu-{architecture}-{name}"
                    straight, resumed = self.root / (tag + "-straight"), self.root / (tag + "-resumed")
                    train(config, self.data, straight)
                    train(config, self.data, resumed, stop_after_steps=1)
                    train(config, self.data, resumed, resume=True)
                    a, _ = read_checkpoint(straight)
                    b, _ = read_checkpoint(resumed)
                    for key in ("model", "optimizer", "sampler", "scheduler", "fisher_rng"):
                        assert_nested(self, a[key], b[key], approximate=True)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
