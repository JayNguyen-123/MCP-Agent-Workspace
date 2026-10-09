"""Hardened primitives for running untrusted, model-generated Python.

Everything here is transport-agnostic so it can be unit-tested without MCP.

Defence in depth (outer layers are configured in docker-compose.yml):
  1. Separate container, non-root user, read-only root FS, no access to app secrets,
     attached only to an internal network with no internet egress.
  2. Per-session working directories, so concurrent agent threads cannot see or
     clobber each other's files.
  3. Each execution runs with a scrubbed environment (no API keys / DB URLs leak),
     in its own process group, under CPU / memory / file-size / process rlimits,
     with a wall-clock timeout and a hard cap on captured output.
"""

from __future__ import annotations

import asyncio
import os
import re
import resource
import shutil
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}(/[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}){0,4}$")


class SandboxError(ValueError):
    """Raised for requests that are rejected before anything is executed."""


@dataclass(frozen=True)
class SandboxLimits:
    timeout_s: float = 10.0
    cpu_s: int = 10
    memory_bytes: int = 512 * 1024 * 1024
    max_file_bytes: int = 10 * 1024 * 1024
    max_processes: int = 64
    max_open_files: int = 256
    max_output_bytes: int = 16 * 1024
    max_code_bytes: int = 200 * 1024

    @classmethod
    def from_env(cls) -> SandboxLimits:
        d = cls()
        return cls(
            timeout_s=float(os.getenv("SANDBOX_TIMEOUT_S", d.timeout_s)),
            cpu_s=int(os.getenv("SANDBOX_CPU_S", d.cpu_s)),
            memory_bytes=int(os.getenv("SANDBOX_MEMORY_BYTES", d.memory_bytes)),
            max_file_bytes=int(os.getenv("SANDBOX_MAX_FILE_BYTES", d.max_file_bytes)),
            max_processes=int(os.getenv("SANDBOX_MAX_PROCESSES", d.max_processes)),
            max_open_files=int(os.getenv("SANDBOX_MAX_OPEN_FILES", d.max_open_files)),
            max_output_bytes=int(os.getenv("SANDBOX_MAX_OUTPUT_BYTES", d.max_output_bytes)),
            max_code_bytes=int(os.getenv("SANDBOX_MAX_CODE_BYTES", d.max_code_bytes)),
        )


@dataclass(frozen=True)
class ExecResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False

    def render(self) -> str:
        status = "TIMED OUT" if self.timed_out else f"exit code {self.exit_code}"
        parts = [f"Execution finished ({status})."]
        if self.stdout:
            parts.append(f"--- stdout ---\n{self.stdout}")
        if self.stderr:
            parts.append(f"--- stderr ---\n{self.stderr}")
        if self.truncated:
            parts.append("[output truncated]")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def session_dir(root: Path, session_id: str) -> Path:
    """Return (and create) the private working directory for one agent thread."""
    if not _SESSION_ID_RE.fullmatch(session_id or ""):
        raise SandboxError("Invalid session id.")
    root = root.resolve()
    path = (root / session_id).resolve()
    if path.parent != root:
        raise SandboxError("Invalid session id.")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def safe_path(base: Path, filename: str) -> Path:
    """Resolve ``filename`` inside ``base`` or raise.

    Uses a component-wise containment check after resolving symlinks. The original
    ``str.startswith`` check accepted siblings such as ``/app/sandbox_evil``.
    """
    if not filename or "\x00" in filename or not _FILENAME_RE.fullmatch(filename):
        raise SandboxError(f"Rejected file name: {filename!r}")
    base = base.resolve()
    target = (base / filename).resolve()
    if target == base or not target.is_relative_to(base):
        raise SandboxError("Path escapes the sandbox workspace.")
    return target


# --------------------------------------------------------------------------- #
# Process execution
# --------------------------------------------------------------------------- #
def _scrubbed_env(workdir: Path) -> dict[str, str]:
    # Never inherit the parent environment: it may hold API keys or DSNs.
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }


def _make_preexec(limits: SandboxLimits):
    def _apply() -> None:  # runs in the child between fork() and exec()
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_s, limits.cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_file_bytes, limits.max_file_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (limits.max_open_files, limits.max_open_files))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
        os.umask(0o077)

    return _apply


async def _read_capped(stream: asyncio.StreamReader, cap: int, sink: bytearray) -> bool:
    """Read the whole stream, keeping at most ``cap`` bytes. Returns True if truncated."""
    truncated = False
    while chunk := await stream.read(65536):
        room = cap - len(sink)
        if room > 0:
            sink.extend(chunk[:room])
        if len(chunk) > room:
            truncated = True
    return truncated


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def run_process(
    argv: list[str],
    *,
    cwd: Path,
    limits: SandboxLimits,
    timeout_s: float | None = None,
    env: dict[str, str] | None = None,
    apply_rlimits: bool = True,
) -> ExecResult:
    timeout_s = limits.timeout_s if timeout_s is None else timeout_s
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=env if env is not None else _scrubbed_env(cwd),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,  # own process group so we can kill grandchildren
        preexec_fn=_make_preexec(limits) if apply_rlimits else None,  # noqa: PLW1509
    )
    out, err = bytearray(), bytearray()
    timed_out = False
    try:
        trunc = await asyncio.wait_for(
            asyncio.gather(
                _read_capped(proc.stdout, limits.max_output_bytes, out),
                _read_capped(proc.stderr, limits.max_output_bytes, err),
                proc.wait(),
            ),
            timeout=timeout_s,
        )
        truncated = bool(trunc[0] or trunc[1])
    except TimeoutError:
        timed_out, truncated = True, False
        _kill_group(proc)
        await proc.wait()
    finally:
        if proc.returncode is None:
            _kill_group(proc)
            await proc.wait()
    return ExecResult(
        exit_code=None if timed_out else proc.returncode,
        stdout=out.decode("utf-8", errors="replace"),
        stderr=err.decode("utf-8", errors="replace"),
        timed_out=timed_out,
        truncated=truncated,
    )


# --------------------------------------------------------------------------- #
# High-level operations used by the MCP tools
# --------------------------------------------------------------------------- #
def sandbox_python() -> str:
    """Interpreter used to run agent code (a dedicated venv in production)."""
    return os.getenv("SANDBOX_PYTHON", sys.executable)


def write_source(workdir: Path, filename: str, code: str, limits: SandboxLimits) -> Path:
    if not filename.endswith(".py"):
        filename += ".py"
    if len(code.encode("utf-8")) > limits.max_code_bytes:
        raise SandboxError(f"Code exceeds the {limits.max_code_bytes}-byte limit.")
    path = safe_path(workdir, filename)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # O_NOFOLLOW: refuse to write through a symlink the agent may have planted.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(code)
    return path


async def run_python_file(workdir: Path, path: Path, limits: SandboxLimits) -> ExecResult:
    # -I: isolated mode (ignores PYTHON* env vars and the user site dir).
    return await run_process([sandbox_python(), "-I", str(path)], cwd=workdir, limits=limits)


async def lint_python_file(workdir: Path, filename: str, limits: SandboxLimits) -> ExecResult:
    path = safe_path(workdir, filename)
    if not path.is_file():  # noqa: ASYNC240 - single stat() on local disk
        raise SandboxError(f"File {filename!r} does not exist in this session.")
    ruff = _ruff_binary()
    rel = str(path.relative_to(workdir.resolve()))  # noqa: ASYNC240
    # --isolated ignores any ruff config the agent may have written into the workspace.
    argv = [ruff, "check", "--isolated", "--no-cache", "--output-format=concise", rel]
    return await run_process(argv, cwd=workdir, limits=limits, apply_rlimits=False)


def _ruff_binary() -> str:
    try:
        from ruff.__main__ import find_ruff_bin  # installed via the `sandbox` extra

        return str(find_ruff_bin())
    except (ImportError, FileNotFoundError):
        found = shutil.which("ruff")
        if not found:
            raise SandboxError("Linter is not installed in the sandbox image.") from None
        return found


def package_allowlist() -> set[str] | None:
    raw = os.getenv("SANDBOX_PACKAGE_ALLOWLIST", "").strip()
    if not raw:
        return None
    return {canonicalize_name(p.strip()) for p in raw.split(",") if p.strip()}


def validate_requirement(spec: str, allowlist: set[str] | None = None) -> str:
    """Accept only a plain PEP 508 requirement from the index: no URLs, paths or flags."""
    spec = (spec or "").strip()
    if not spec or spec.startswith("-") or len(spec) > 200 or any(c in spec for c in "\n\r\t /\\@;"):
        raise SandboxError(f"Rejected package specification: {spec!r}")
    try:
        req = Requirement(spec)
    except InvalidRequirement as exc:
        raise SandboxError(f"Invalid package specification: {exc}") from None
    if req.url or req.marker:
        raise SandboxError("Direct URLs and environment markers are not allowed.")
    if allowlist is not None and canonicalize_name(req.name) not in allowlist:
        raise SandboxError(f"Package {req.name!r} is not on the sandbox allowlist.")
    return str(req)


async def install_package(spec: str, limits: SandboxLimits) -> ExecResult:
    """Install into the *sandbox* interpreter only, from wheels only.

    The original used ``uv add`` which rewrote the API service's own pyproject.toml
    and lockfile, and allowed sdists whose setup.py runs arbitrary code at build time.
    """
    req = validate_requirement(spec, package_allowlist())
    argv = [
        sandbox_python(), "-m", "pip", "install",
        "--no-input", "--disable-pip-version-check", "--only-binary=:all:",
        req,
    ]  # fmt: skip
    scratch = Path(tempfile.gettempdir())
    env = _scrubbed_env(scratch)
    for key in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST", "HTTPS_PROXY", "NO_PROXY"):
        if os.environ.get(key):  # compose passes unset values as "", which would break pip
            env[key] = os.environ[key]
    return await run_process(argv, cwd=scratch, limits=limits, timeout_s=180, env=env, apply_rlimits=False)
