"""Prepare immutable FineWeb-Edu token files with document-level holdouts."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import unicodedata

import numpy as np
from tokenizers import Tokenizer

from artifacts import atomic_json, file_lock, fingerprint, sha256_file

DATASET = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
TOKENIZER_SHA256 = "ac598311c740d555cb6f4f63c1a3cdd4d4cf63b7f61685481512ab6e4910517b"
SPLITS = ("train", "val", "test")


def document_split(text):
    # Whitespace/NFC-equivalent documents always stay in the same split, even
    # across shards. This is not a claim of near-duplicate decontamination.
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    digest = hashlib.sha256(("optimizer-study-v1\0" + normalized).encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10000
    return "val" if bucket < 100 else "test" if bucket < 200 else "train"


def source_batches(recipe, filename):
    import pyarrow.parquet as pq
    if recipe["dataset"] == DATASET:
        from huggingface_hub import hf_hub_download
        filename = hf_hub_download(DATASET, filename, repo_type="dataset",
                                   revision=recipe["revision"])
    for batch in pq.ParquetFile(filename).iter_batches(batch_size=128, columns=["text"]):
        yield batch.column("text").to_pylist()


def prepare(output, tokenizer_path, recipe):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(tokenizer_path)
    if sha256_file(tokenizer_path) != recipe["tokenizer_sha256"]:
        raise ValueError("Tokenizer checksum does not match the data recipe")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("<eos>")
    vocab_size = tokenizer.get_vocab_size()
    if eos is None or max(tokenizer.get_vocab().values()) >= 65536:
        raise ValueError("A uint16-compatible tokenizer with <eos> is required")
    if set(recipe["budgets"]) != set(SPLITS) or any(
            type(v) is not int or v < 2 for v in recipe["budgets"].values()):
        raise ValueError("Each split needs a positive integer budget of at least two tokens")
    recipe_id = fingerprint(recipe)
    with file_lock(output / ".prepare.lock"):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["recipe_id"] != recipe_id:
                raise ValueError("Existing dataset has a different recipe; choose a new output directory")
            verify_data(output)
            print(f"Dataset already complete: {output}", flush=True)
            return manifest
        state_path = output / "prepare-state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["recipe_id"] != recipe_id:
                raise ValueError("Partial dataset has a different recipe; choose a new output directory")
        else:
            if any((output / f"{s}.bin").exists() for s in SPLITS):
                raise ValueError("Token files exist without preparation state; refusing to overwrite")
            state = dict(recipe_id=recipe_id, file_index=0, row_index=0,
                         tokens={s: 0 for s in SPLITS}, documents={s: 0 for s in SPLITS})
            atomic_json(state_path, state)
        with ExitStack() as stack:
            streams = {}
            for split in SPLITS:
                path = output / f"{split}.bin"
                stream = stack.enter_context(open(path, "r+b" if path.exists() else "w+b"))
                committed = state["tokens"][split] * 2
                if os.fstat(stream.fileno()).st_size < committed:
                    raise ValueError(f"{path} is shorter than its committed preparation state")
                stream.truncate(committed)
                stream.seek(committed)
                streams[split] = stream

            def commit():
                for stream in streams.values():
                    stream.flush()
                    os.fsync(stream.fileno())
                atomic_json(state_path, state)
                print("Prepared " + ", ".join(f"{s}={state['tokens'][s]:,}" for s in SPLITS), flush=True)

            finished = lambda: all(state["tokens"][s] >= recipe["budgets"][s] for s in SPLITS)
            batches = 0
            for index in range(state["file_index"], len(recipe["files"])):
                if finished():
                    break
                resume_row = state["row_index"] if index == state["file_index"] else 0
                state["file_index"] = index
                row = 0
                for texts in source_batches(recipe, recipe["files"][index]):
                    if row + len(texts) <= resume_row:
                        row += len(texts)
                        continue
                    start = max(0, resume_row - row)
                    selected = []
                    for text in texts[start:]:
                        if isinstance(text, str) and text.strip():
                            split = document_split(text)
                            if state["tokens"][split] < recipe["budgets"][split]:
                                selected.append((split, text))
                    encoded = tokenizer.encode_batch([t for _, t in selected], add_special_tokens=False)
                    for (split, _), enc in zip(selected, encoded):
                        remaining = recipe["budgets"][split] - state["tokens"][split]
                        if remaining <= 0:
                            continue
                        ids = np.asarray((enc.ids + [eos])[:remaining], dtype="<u2")
                        ids.tofile(streams[split])
                        state["tokens"][split] += len(ids)
                        state["documents"][split] += 1
                    row += len(texts)
                    state["row_index"] = row
                    batches += 1
                    if batches % 16 == 0:
                        commit()
                    if finished():
                        break
                if not finished():
                    state["file_index"] = index + 1
                    state["row_index"] = 0
                commit()
            if not finished():
                raise RuntimeError("Source exhausted before reaching every split budget; partial state retained")
        # Preserve the original bytes, since tokenizers.save can reformat JSON.
        with open(output / "tokenizer.json", "wb") as stream:
            stream.write(tokenizer_path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        manifest = dict(version=1, recipe=recipe, recipe_id=recipe_id, dtype="<u2",
                        vocab_size=vocab_size, eos_id=eos, documents=state["documents"],
                        splits={s: dict(file=f"{s}.bin", tokens=state["tokens"][s],
                                        sha256=sha256_file(output / f"{s}.bin")) for s in SPLITS})
        manifest["data_id"] = fingerprint(manifest)
        atomic_json(manifest_path, manifest)
        print(f"Ready: {output / 'manifest.json'}", flush=True)
        return manifest


def verify_data(directory, checksums=True):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = dict(manifest)
    data_id = expected.pop("data_id")
    if fingerprint(expected) != data_id or manifest["dtype"] != "<u2":
        raise ValueError("Dataset manifest fingerprint or dtype is invalid")
    if sha256_file(directory / "tokenizer.json") != manifest["recipe"]["tokenizer_sha256"]:
        raise ValueError("Prepared tokenizer checksum differs")
    for split in SPLITS:
        item = manifest["splits"][split]
        if item["file"] != f"{split}.bin":
            raise ValueError("Unexpected data filename")
        path = directory / item["file"]
        if path.stat().st_size != item["tokens"] * 2:
            raise ValueError(f"Wrong token-file size: {path}")
        if checksums and sha256_file(path) != item["sha256"]:
            raise ValueError(f"Wrong token-file checksum: {path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/fineweb-edu-512m")
    parser.add_argument("--tokenizer", default=str(Path(__file__).parent / "tokenizer/tokenizer.json"))
    parser.add_argument("--train-tokens", type=int, default=536870912)
    parser.add_argument("--val-tokens", type=int, default=1048576)
    parser.add_argument("--test-tokens", type=int, default=1048576)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_data(args.output), indent=2))
        return
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(DATASET, revision=args.revision)
    files = sorted(x.rfilename for x in info.siblings
                   if x.rfilename.startswith("sample/10BT/") and x.rfilename.endswith(".parquet"))
    if not files:
        raise RuntimeError("Pinned FineWeb-Edu revision has no sample/10BT parquet files")
    random.Random(3619).shuffle(files)
    recipe = dict(dataset=DATASET, subset="sample-10BT", revision=info.sha,
                  files=files, file_shuffle_seed=3619, split_rule="sha256-nfc-whitespace-v1-98/1/1",
                  tokenizer_sha256=TOKENIZER_SHA256,
                  budgets=dict(train=args.train_tokens, val=args.val_tokens, test=args.test_tokens))
    prepare(args.output, args.tokenizer, recipe)


if __name__ == "__main__":
    main()
