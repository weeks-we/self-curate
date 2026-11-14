"""Shell-free official CLI invocation with logs, timeouts and process-group cleanup."""

from datetime import datetime, timezone
from pathlib import Path
import os
import signal
import subprocess
import time
from self_curate.training.recovery import atomic_json


def execute(argv, *, cwd, output, timeout, env=None, secret_values=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).isoformat()
    atomic_json(
        output / "command.json",
        {
            "argv": [__import__("functools").reduce(lambda value, secret: value.replace(secret, "<redacted>") if secret else value, secret_values, str(arg)) for arg in argv],
            "cwd": str(cwd),
            "started_at": stamp,
            "timeout_seconds": timeout,
            "environment_values_logged": False,
        },
    )
    timed_out = False
    with (
        (output / "stdout.log").open("wb") as stdout,
        (output / "stderr.log").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            timed_out = True
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
    value = {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "wall_seconds": time.perf_counter() - started,
        "started_at": stamp,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output / "timing.json", value)
    return value
