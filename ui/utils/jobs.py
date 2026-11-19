"""Running long tasks (training steps) with a live log and a stop button.

Pipeline scripts read arguments from `sys.argv` at module level,
so they are launched as a separate process — same as the official RVC WebUI.
"""

import io
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque

import gradio as gr

# Repository root (ui/utils/jobs.py -> ui -> project root): subprocesses run
# with this working directory so `python -m programs.rvc.training...` resolves.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAIL_LINES = 200  # How many last lines of the log to show

_lock = threading.Lock()
_process = None
_process_group_id = None
_stop_requested = False


def busy() -> bool:
    with _lock:
        proc = _process
        group_id = _process_group_id
    if proc is None:
        return False
    if proc.poll() is None:
        return True
    if os.name != "nt" and group_id is not None:
        try:
            os.killpg(group_id, 0)
            return True
        except (OSError, ProcessLookupError):
            pass
    return False


def command(*args) -> list:
    """Command to launch a project module with the same interpreter as the interface."""
    return [sys.executable or "python", "-m", *[str(arg) for arg in args]]


def _kill_process_tree(proc: subprocess.Popen, sig: int, process_group_id=None) -> None:
    """Stops the process and its descendants, even if the leader has already exited."""
    if os.name == "nt":
        # Popen.terminate() only kills the parent; taskkill /T is needed
        # to also stop multiprocessing/DataLoader workers along with it.
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        # Do not call os.getpgid(proc.pid) here: after the leader exits this is
        # impossible, even though its child processes may still be alive.
        os.killpg(process_group_id if process_group_id is not None else proc.pid, sig)
    except (OSError, ProcessLookupError):
        pass


def request_stop() -> str:
    """Handler for the "Stop" button. Called with queue=False."""
    global _process, _stop_requested
    with _lock:
        _stop_requested = True
        proc = _process
        process_group_id = _process_group_id
    if proc is None:
        return "No process is running."

    _kill_process_tree(proc, signal.SIGTERM, process_group_id)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

    # We don't consider the task stopped just because the parent Python
    # process exited: DataLoader workers may outlive it and hold RAM.
    _kill_process_tree(proc, signal.SIGKILL if os.name != "nt" else signal.SIGTERM, process_group_id)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return "Failed to stop the process — check for remaining child processes."
    return "The process and its child processes have been stopped."


def _release_inference_cache() -> None:
    """Frees the inference cache: the training process needs all the GPU memory.

    The import is lazy — `programs.rvc.infer.infer` pulls in torch, and the interface starts without it.
    """
    try:
        from programs.rvc.infer.infer import release_pipeline
    except Exception:  # noqa: BLE001 — without torch/dependencies there is nothing to unload
        return
    released = release_pipeline()
    if released:
        print(f"[i] Model '{released}' unloaded from memory before starting the task.", flush=True)


def _spawn(cmd: list) -> subprocess.Popen:
    global _process, _process_group_id
    _release_inference_cache()
    kwargs = {
        "cwd": PROJECT_ROOT,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with _lock:
        _process = subprocess.Popen(cmd, **kwargs)  # noqa: S603 — the command is built from constants
        _process_group_id = _process.pid if os.name != "nt" else None
        # Text wrapper without \r translation: this is live tqdm progress, not a line ending.
        _process.stdout = io.TextIOWrapper(_process.stdout, encoding="utf-8", errors="replace", newline="")
        return _process


def run_job(title: str, commands: list, prefix: str = ""):
    """Generator for the start button: runs commands in sequence, yields the log."""
    global _process, _process_group_id, _stop_requested
    if busy():
        raise gr.Error("Another task is still running — wait for it or stop it.")

    if commands and isinstance(commands[0], str):
        commands = [commands]
    with _lock:
        _stop_requested = False

    lines = deque(maxlen=TAIL_LINES)
    lines.append(f"▶ {title}")
    if prefix:
        lines.extend(prefix.split("\n"))
    yield "\n".join(lines)

    try:
        for cmd in commands:
            with _lock:
                if _stop_requested:
                    break
            lines.append(f"$ {' '.join(cmd)}")
            proc = _spawn(cmd)
            last_yield = 0.0
            buffer = ""
            live = ""
            while True:
                # Read character by character: tqdm sends only \r without \n for minutes,
                # and line-by-line reading hangs silently during that time.
                chunk = proc.stdout.read(1)
                if chunk:
                    if chunk in "\r\n":
                        text = buffer.strip()
                        buffer = ""
                        if chunk == "\n":
                            # Jupyter/Colab does not redraw \r in console output:
                            # there we print the completed tqdm bar once, on LF.
                            # In a regular TTY we finish the already-updating bar line by line.
                            console_text = live or text
                            if console_text:
                                lines.append(console_text)
                                if live and sys.stdout.isatty():
                                    print(flush=True)
                                else:
                                    print(console_text, flush=True)
                            live = ""
                        elif chunk == "\r":
                            if text:
                                live = text
                                if sys.stdout.isatty():
                                    print("\r" + live, end="", flush=True)
                        elif text:
                            live = text
                    else:
                        buffer += chunk
                elif proc.poll() is not None:
                    break
                else:
                    time.sleep(0.05)
                if time.monotonic() - last_yield >= 0.3:
                    last_yield = time.monotonic()
                    yield "\n".join([*lines, live] if live else lines)
            tail = buffer.strip()
            if tail:
                print(tail, flush=True)
                lines.append(tail)
            if live:
                if sys.stdout.isatty():
                    print(flush=True)
                else:
                    print(live, flush=True)
                lines.append(live)
            proc.wait()
            with _lock:
                stopped = _stop_requested
            if stopped:
                lines.append("⏹ Stopped by user.")
                yield "\n".join(lines)
                return
            if proc.returncode != 0:
                lines.append(f"❌ Error (code {proc.returncode}). See the log above.")
                yield "\n".join(lines)
                return
    finally:
        # We don't reset the flag: the next run will set it itself, there are no readers between runs.
        with _lock:
            _process = None
            _process_group_id = None

    lines.append("✓ Done.")
    yield "\n".join(lines)
