"""Check GPU selection, cleanup and failure propagation without a GPU or Docker."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        shutil.copy(Path(__file__).with_name("run_ml2.sh"), root)
        binaries = root / "bin"
        binaries.mkdir()
        docker = binaries / "docker"
        docker.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['TEST_CALL_LOG'], 'a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == 'run':
    sys.exit(int(os.environ['TEST_DOCKER_EXIT']))
print('{}')
""")
        for name in ("git", "nvidia-smi"):
            (binaries / name).write_text("#!/bin/sh\nprintf 'test metadata\\n'\n")
        for binary in binaries.iterdir():
            binary.chmod(0o755)
        for code in (0, 7):
            log = root / f"calls-{code}.jsonl"
            env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                       TEST_CALL_LOG=str(log), TEST_DOCKER_EXIT=str(code))
            result = subprocess.run(["bash", str(root / "run_ml2.sh"), "GPU-test", "smoke"],
                                    env=env, capture_output=True, text=True)
            assert result.returncode == code, result.stderr
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            run = next(call for call in calls if call[0] == "run")
            assert run[run.index("--gpus") + 1] == "device=GPU-test"
            assert "--rm" in run and "timeout" in run and "10m" in run
            assert calls[-1] == ["stop", run[run.index("--name") + 1]]
        bad = subprocess.run(["bash", str(root / "run_ml2.sh"), "all"],
                             env=env, capture_output=True, text=True)
        assert bad.returncode == 2
    print("Runner check passed: one selected GPU, bounded runtime, cleanup and exit status.")


if __name__ == "__main__":
    main()
