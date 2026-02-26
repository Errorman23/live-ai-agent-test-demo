from __future__ import annotations

import logging
import shlex
import signal
import socket
import subprocess
import threading
import time
import os
import shutil
from pathlib import Path
from typing import Any

from app.config import Settings

# Persistent Promptfoo viewer process manager.
# Responsibilities:
# - start/stop/restart the viewer subprocess on a fixed port
# - run health probes and expose lightweight diagnostics for UI/API
# - keep viewer lifecycle independent from per-batch eval execution
# Boundaries:
# - campaign execution itself is handled by testing/runner.py

logger = logging.getLogger(__name__)


def _port_open(host: str, port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def cleanup_orphan_promptfoo_eval_processes(project_root: Path) -> dict[str, Any]:
    """
    Best-effort cleanup for orphaned promptfoo eval/redteam-eval subprocesses.

    This prevents stale background jobs from continuing to call /api/v1/run
    after backend restarts.
    """
    marker_path = str((project_root / "backend/data/artifacts/promptfoo").resolve())
    summary: dict[str, Any] = {
        "terminated": [],
        "killed": [],
        "errors": [],
    }
    try:
        listing = subprocess.run(
            ["ps", "axww", "-o", "pid=,command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append(f"ps failed: {exc}")
        return summary

    for raw in listing.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_token, command = parts
        if not pid_token.isdigit():
            continue
        pid = int(pid_token)
        if pid == os.getpid():
            continue
        command_lower = command.lower()
        if "promptfoo" not in command_lower:
            continue
        if " view " in command_lower:
            continue
        is_eval = (" redteam eval " in command_lower) or (" eval -c " in command_lower)
        if not is_eval:
            continue
        if marker_path not in command and "backend/data/artifacts/promptfoo" not in command:
            continue

        try:
            os.kill(pid, signal.SIGTERM)
            summary["terminated"].append(pid)
        except ProcessLookupError:
            continue
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"SIGTERM pid={pid}: {exc}")

    if summary["terminated"]:
        time.sleep(1.0)
        for pid in list(summary["terminated"]):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except Exception:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                summary["killed"].append(pid)
            except ProcessLookupError:
                continue
            except Exception as exc:  # noqa: BLE001
                summary["errors"].append(f"SIGKILL pid={pid}: {exc}")

    if summary["terminated"] or summary["killed"] or summary["errors"]:
        logger.warning(
            "Promptfoo orphan cleanup summary: terminated=%s killed=%s errors=%s",
            summary["terminated"],
            summary["killed"],
            summary["errors"],
        )
    return summary


class PromptfooServiceManager:
    def __init__(self, settings: Settings, *, project_root: Path) -> None:
        self.settings = settings
        self.project_root = project_root
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._log_file: Any | None = None
        self._last_error: str | None = None
        self._last_started_at: float | None = None
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def ui_url(self) -> str:
        return f"http://127.0.0.1:{self.settings.promptfoo_port}"

    @property
    def output_dir(self) -> Path:
        output_root = Path(self.settings.promptfoo_output_dir)
        if not output_root.is_absolute():
            output_root = self.project_root / output_root
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root

    @property
    def log_path(self) -> Path:
        path = Path(self.settings.promptfoo_log_path)
        if not path.is_absolute():
            path = self.project_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def start(self) -> None:
        if not self.settings.promptfoo_enabled:
            return
        with self._lock:
            if self._is_alive_locked():
                return
            self._last_error = None
            cmd = shlex.split(self.settings.promptfoo_command) + [
                "view",
                "--port",
                str(self.settings.promptfoo_port),
                "--yes",
                str(self.output_dir),
            ]
            log_file = self.log_path.open("a", encoding="utf-8")
            self._log_file = log_file
            try:
                env = dict(os.environ)
                env["PROMPTFOO_DISABLE_TELEMETRY"] = "1"
                env["DO_NOT_TRACK"] = "1"
                # Stabilize viewer startup across mixed local Node versions.
                parts = shlex.split(self.settings.promptfoo_command)
                if parts and Path(parts[0]).name == "npx":
                    resolved_npx = (
                        str(Path(parts[0]).expanduser())
                        if "/" in parts[0]
                        else shutil.which(parts[0], path=env.get("PATH"))
                    )
                    if resolved_npx:
                        npx_dir = str(Path(resolved_npx).parent)
                        current_path = env.get("PATH", "")
                        env["PATH"] = f"{npx_dir}:{current_path}" if current_path else npx_dir
                        sibling_node = Path(npx_dir) / "node"
                        node_bin = str(sibling_node) if sibling_node.exists() else shutil.which("node", path=env["PATH"])
                        if node_bin:
                            try:
                                completed = subprocess.run(
                                    [node_bin, "-e", "process.stdout.write(process.versions.node.split('.')[0])"],
                                    check=False,
                                    capture_output=True,
                                    text=True,
                                    timeout=5,
                                )
                                token = (completed.stdout or "").strip()
                                if completed.returncode == 0 and token.isdigit():
                                    cache_dir = self.project_root / ".run" / f"npm-cache-node{int(token)}"
                                    cache_dir.mkdir(parents=True, exist_ok=True)
                                    env["npm_config_cache"] = str(cache_dir)
                            except Exception:
                                pass
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                self._last_started_at = time.time()
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"Promptfoo viewer failed to start: {exc}"
                self._proc = None
                log_file.close()
                self._log_file = None
                return

            # Give promptfoo viewer a short window to bind.
            for _ in range(20):
                if _port_open("127.0.0.1", self.settings.promptfoo_port):
                    break
                if self._proc.poll() is not None:
                    self._last_error = (
                        "Promptfoo viewer exited during startup. "
                        f"Return code={self._proc.returncode}"
                    )
                    self._proc = None
                    if self._log_file is not None:
                        self._log_file.close()
                        self._log_file = None
                    break
                time.sleep(0.2)

            if self._monitor_thread is None or (not self._monitor_thread.is_alive()):
                self._stop_event.clear()
                self._monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    name="promptfoo-monitor",
                    daemon=True,
                )
                self._monitor_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            proc = self._proc
            self._proc = None
            log_file = self._log_file
            self._log_file = None
        if proc is None:
            if log_file is not None:
                log_file.close()
            return
        try:
            proc.terminate()
            proc.wait(timeout=4)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
        if log_file is not None:
            log_file.close()

    def restart(self) -> None:
        self.stop()
        self.start()

    def ensure_running(self) -> None:
        if not self.settings.promptfoo_enabled:
            return
        if not self.health()["healthy"]:
            self.start()

    def health(self) -> dict[str, Any]:
        with self._lock:
            pid = self._proc.pid if self._proc is not None else None
            proc_alive = self._is_alive_locked()
            started_at = self._last_started_at
            last_error = self._last_error
        port_open = _port_open("127.0.0.1", self.settings.promptfoo_port)
        return {
            "enabled": self.settings.promptfoo_enabled,
            "healthy": bool(proc_alive and port_open),
            "process_alive": proc_alive,
            "port_open": port_open,
            "pid": pid,
            "ui_url": self.ui_url if port_open else None,
            "last_error": last_error,
            "last_started_at_unix": started_at,
            "log_path": str(self.log_path),
        }

    def log_tail(self, lines: int = 200) -> dict[str, Any]:
        lines = max(1, min(lines, 2000))
        path = self.log_path
        if not path.exists():
            return {"log_path": str(path), "lines": []}
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"log_path": str(path), "lines": data[-lines:]}

    def _is_alive_locked(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(5)
            if not self.settings.promptfoo_enabled:
                continue
            health = self.health()
            if health["healthy"]:
                continue
            with self._lock:
                if self._proc is not None and self._proc.poll() is not None:
                    self._last_error = (
                        "Promptfoo viewer exited unexpectedly. "
                        f"Return code={self._proc.returncode}"
                    )
                    self._proc = None
                    if self._log_file is not None:
                        self._log_file.close()
                        self._log_file = None
            self.start()
