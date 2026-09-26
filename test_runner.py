"""Check GPU selection, cleanup and failure propagation without a GPU or Docker."""
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile
import time


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
    check_training_launcher()
    if shutil.which("tmux"):
        check_tmux_interrupt()
    print("Runner check passed: one selected GPU, bounded runtime, cleanup and exit status.")


def check_training_launcher():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        shutil.copy(Path(__file__).with_name("launch_study.sh"), root)
        (root / ".venv/bin").mkdir(parents=True)
        binaries = root / "bin"
        binaries.mkdir()
        (root / ".venv/bin/python").write_text("""#!/usr/bin/env python3
import json, sys
if sys.argv[1] == '-':
    sys.stdin.read()
else:
    open('arguments.json', 'w').write(json.dumps(sys.argv[1:]))
    print('controlled queue failure')
    sys.exit(7)
""")
        (binaries / "tmux").write_text("""#!/usr/bin/env python3
import os, subprocess, sys
args = sys.argv[1:]
if args[0] == 'has-session':
    sys.exit(0 if os.environ.get('TEST_EXISTING_SESSION') else 1)
if args[0] == 'new-session':
    subprocess.run(args[args.index('bash'):])
""")
        for binary in (root / ".venv/bin/python", binaries / "tmux"):
            binary.chmod(0o755)
        (root / "plan with spaces.json").write_text('{}')
        env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"], STUDY_HOURS="0.5")
        result = subprocess.run(["bash", str(root / "launch_study.sh"), "--retry-failed", "plan with spaces.json", "GPU-test"],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        args = json.loads((root / "arguments.json").read_text())
        assert args[args.index('--plan') + 1] == 'plan with spaces.json'
        assert args[args.index('--gpus') + 1] == 'GPU-test'
        assert args[args.index('--hours') + 1] == '0.5'
        assert '--retry-failed' in args
        assert (root / 'results/queue.exit').read_text().strip() == '7'
        result = subprocess.run(["bash", str(root / "launch_study.sh"), "plan with spaces.json", "GPU-test"],
                                env=dict(env, TEST_EXISTING_SESSION="1"), capture_output=True, text=True)
        assert result.returncode == 2 and 'already exists' in result.stderr


def check_tmux_interrupt():
    """Check cache setup and interrupts against an already-running tmux server."""
    with tempfile.TemporaryDirectory() as directory:
        installation = Path(directory)
        root = installation / "repo"
        root.mkdir()
        cache = installation / "cache with spaces"
        (installation / "env.sh").write_text('export TILELANG_CACHE_DIR=' + shlex.quote(str(cache)) + '\n')
        shutil.copy(Path(__file__).with_name("launch_study.sh"), root)
        (root / ".venv/bin").mkdir(parents=True)
        (root / "bin").mkdir()
        tmux = [shutil.which("tmux"), "-S", str(root / "tmux.sock")]
        wrapper = root / "bin/tmux"
        wrapper.write_text('#!/bin/sh\nexec ' + shlex.join(tmux) + ' "$@"\n')
        python = root / ".venv/bin/python"
        python.write_text("""#!/usr/bin/env python3
from pathlib import Path
import os, signal, sys, time
assert os.environ.get('TILELANG_CACHE_DIR') == str(Path.cwd().parent / 'cache with spaces'), 'scratch cache environment lost'
if sys.argv[1] == '-':
    sys.stdin.read()
    sys.exit(0)
stopped = False
def stop(signum, frame):
    global stopped
    stopped = True
signal.signal(signal.SIGINT, stop)
Path('ready').touch()
while not stopped:
    time.sleep(.05)
print('STOP SAVED', flush=True)
""")
        for path in (wrapper, python):
            path.chmod(0o755)
        (root / "plan.json").write_text('{}')
        env = dict(os.environ, PATH=str(root / 'bin') + os.pathsep + os.environ['PATH'])
        try:
            subprocess.run(tmux + ['new-session', '-d', '-s', 'older-session'], env=env, check=True)
            subprocess.run(tmux + ['set-environment', '-g', '-u', 'TILELANG_CACHE_DIR'], check=True)
            launched = subprocess.run(['bash', str(root / 'launch_study.sh'), 'plan.json', 'GPU-test'],
                                      env=env, capture_output=True, text=True, timeout=15)
            assert launched.returncode == 0, launched.stderr
            deadline = time.monotonic() + 10
            while not (root / 'ready').exists() and time.monotonic() < deadline:
                time.sleep(.05)
            assert (root / 'ready').exists(), (root / 'results/queue.log').read_text()
            subprocess.run(tmux + ['send-keys', '-t', 'optimizer-training:0.0', 'C-c'], check=True)
            while not (root / 'results/queue.exit').exists() and time.monotonic() < deadline:
                time.sleep(.05)
            assert (root / 'results/queue.exit').read_text().strip() == '0'
            assert 'STOP SAVED' in (root / 'results/queue.log').read_text()
        finally:
            subprocess.run(tmux + ['kill-server'], capture_output=True)


if __name__ == "__main__":
    main()
