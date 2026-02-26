from __future__ import annotations

import os
import re
import signal
import shlex
import shutil
import subprocess
from pathlib import Path


def compact_process_output(text: str, limit: int = 220) -> str:
    if not text:
        return ""
    without_ansi = re.sub(r"\x1b\[[0-9;]*m", "", text)
    compact = " ".join(without_ansi.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def clean_output(text: str) -> str:
    if not text:
        return ""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split()).strip()


def is_email_verification_block(stdout: str, stderr: str) -> bool:
    combined = clean_output(f"{stdout}\n{stderr}").lower()
    return bool(re.search(r"email\s+verification\s+required", combined))


def js_assert(condition: str) -> dict[str, str]:
    return {
        "type": "javascript",
        "value": (
            "(() => { try { const o = typeof output === 'string' ? JSON.parse(output) : output; "
            f"return Boolean({condition});"
            " } catch (e) { return false; } })()"
        ),
    }


def run_command_with_timeout(
    cmd: list[str],
    *,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str, bool]:
    """
    Run a command with process-group timeout handling so child processes
    spawned by npm/pf are also terminated on timeout.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=env,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return process.returncode, stdout or "", stderr or "", False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return process.returncode or -9, stdout or "", stderr or "", True


def safe_trace_url(langfuse_host: str | None, trace_id: str | None, project_id: str | None = None) -> str | None:
    _ = project_id
    if not langfuse_host or not trace_id:
        return None
    return f"{langfuse_host.rstrip('/')}/trace/{trace_id}"


def _node_major(node_bin: str) -> int | None:
    try:
        completed = subprocess.run(
            [node_bin, "-e", "process.stdout.write(process.versions.node.split('.')[0])"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    token = (completed.stdout or "").strip()
    if not token.isdigit():
        return None
    major = int(token)
    return major if major > 0 else None


def promptfoo_subprocess_env(
    *,
    promptfoo_command: str | None = None,
    project_root: Path | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    env["PROMPTFOO_DISABLE_TELEMETRY"] = "1"
    env["DO_NOT_TRACK"] = "1"

    # Keep promptfoo's npm/npx execution deterministic across mixed local Node installs.
    # Otherwise stale native addons in ~/.npm/_npx can be loaded by a different Node ABI.
    npx_bin: str | None = None
    if promptfoo_command:
        parts = shlex.split(promptfoo_command)
        if parts:
            head = parts[0]
            if Path(head).name == "npx":
                if "/" in head:
                    npx_bin = str(Path(head).expanduser())
                else:
                    npx_bin = shutil.which(head, path=env.get("PATH"))

    npx_dir: str | None = None
    if npx_bin:
        npx_path = Path(npx_bin)
        if npx_path.exists():
            npx_dir = str(npx_path.parent)
            current_path = env.get("PATH", "")
            env["PATH"] = f"{npx_dir}:{current_path}" if current_path else npx_dir

    node_bin = None
    if npx_dir:
        sibling_node = Path(npx_dir) / "node"
        if sibling_node.exists():
            node_bin = str(sibling_node)
    if node_bin is None:
        node_bin = shutil.which("node", path=env.get("PATH"))

    if node_bin:
        major = _node_major(node_bin)
        if major is not None:
            root = project_root or Path.cwd()
            cache_dir = root / ".run" / f"npm-cache-node{major}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            env["npm_config_cache"] = str(cache_dir)
    return env
