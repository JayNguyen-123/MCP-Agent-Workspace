import asyncio
import os
from pathlib import Path

import pytest

from sandbox_server import executor
from sandbox_server.executor import SandboxError, SandboxLimits

LIMITS = SandboxLimits(timeout_s=3, cpu_s=3, max_output_bytes=1024)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return executor.session_dir(tmp_path / "root", "session_1")


def run(workdir: Path, code: str, limits: SandboxLimits = LIMITS):
    path = executor.write_source(workdir, "main.py", code, limits)
    return asyncio.run(executor.run_python_file(workdir, path, limits))


# ----------------------------------------------------------------- path safety
@pytest.mark.parametrize(
    "name",
    ["../escape.py", "/etc/passwd", "a/../../b.py", "", "x\x00.py", ".hidden.py", "a\\b.py", "a" * 200 + ".py"],
)
def test_safe_path_rejects_escapes(workdir, name):
    with pytest.raises(SandboxError):
        executor.safe_path(workdir, name)


def test_sibling_prefix_bypass_is_blocked(tmp_path):
    """Regression: str.startswith('/x/sandbox') accepted '/x/sandbox_evil/...'."""
    base = tmp_path / "sandbox"
    base.mkdir()
    (tmp_path / "sandbox_evil").mkdir()
    with pytest.raises(SandboxError):
        executor.safe_path(base, "../sandbox_evil/x.py")


def test_symlink_escape_is_blocked(workdir, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (workdir / "link").symlink_to(outside)
    with pytest.raises(SandboxError):
        executor.safe_path(workdir, "link/x.py")


def test_write_refuses_to_follow_symlinked_file(workdir, tmp_path):
    target = tmp_path / "victim.txt"
    target.write_text("original")
    (workdir / "main.py").symlink_to(target)
    with pytest.raises(SandboxError):  # resolves outside the workspace
        executor.write_source(workdir, "main.py", "print(1)", LIMITS)
    assert target.read_text() == "original"


@pytest.mark.parametrize("sid", ["../x", "a/b", "", "x" * 65, "."])
def test_session_ids_validated(tmp_path, sid):
    with pytest.raises(SandboxError):
        executor.session_dir(tmp_path, sid)


def test_sessions_are_isolated(tmp_path):
    a = executor.session_dir(tmp_path, "a")
    b = executor.session_dir(tmp_path, "b")
    assert a != b and a.parent == b.parent == tmp_path.resolve()


# ------------------------------------------------------------------- execution
def test_runs_code_and_captures_output(workdir):
    result = run(workdir, "import sys\nprint(4 + 4)\nprint('warn', file=sys.stderr)")
    assert result.exit_code == 0 and result.stdout.strip() == "8" and "warn" in result.stderr


def test_nonzero_exit_reports_both_streams(workdir):
    result = run(workdir, "print('before')\nraise SystemExit(3)")
    assert result.exit_code == 3 and "before" in result.stdout


def test_secrets_are_not_inherited(workdir, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-leak")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db/x")
    result = run(workdir, "import os; print(sorted(os.environ))")
    assert "ANTHROPIC_API_KEY" not in result.stdout and "DATABASE_URL" not in result.stdout


def test_cwd_is_session_dir(workdir):
    assert run(workdir, "import os; print(os.getcwd())").stdout.strip() == str(workdir)


def test_timeout_kills_process_group(workdir):
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "time.sleep(60)"
    )
    result = run(workdir, code, SandboxLimits(timeout_s=1, cpu_s=5))
    assert result.timed_out and result.exit_code is None


def test_cpu_limit(workdir):
    result = run(workdir, "while True: pass", SandboxLimits(timeout_s=10, cpu_s=1))
    assert not result.timed_out and result.exit_code != 0  # killed by SIGXCPU


def test_memory_limit(workdir):
    limits = SandboxLimits(timeout_s=10, memory_bytes=256 * 1024 * 1024)
    result = run(workdir, "x = bytearray(1024 * 1024 * 1024)\nprint('allocated')", limits)
    assert result.exit_code != 0 and "allocated" not in result.stdout


def test_output_is_capped(workdir):
    result = run(workdir, "print('x' * 1_000_000)")
    assert result.truncated and len(result.stdout) <= LIMITS.max_output_bytes
    assert "[output truncated]" in result.render()


def test_code_size_limit(workdir):
    with pytest.raises(SandboxError):
        executor.write_source(workdir, "big.py", "#" * 1000, SandboxLimits(max_code_bytes=100))


def test_py_suffix_added(workdir):
    assert executor.write_source(workdir, "script", "pass", LIMITS).name == "script.py"


# --------------------------------------------------------------- requirements
@pytest.mark.parametrize("spec", ["pandas", "pandas>=2.0", "requests[socks]==2.32.3", "scikit-learn"])
def test_valid_requirements(spec):
    assert executor.validate_requirement(spec)


@pytest.mark.parametrize(
    "spec",
    [
        "--index-url=http://evil",
        "-e .",
        "foo @ https://evil.example/foo.whl",
        "./local_pkg",
        "foo; python_version<'4'",
        "pkg\n--extra-index-url http://evil",
        "git+https://github.com/x/y",
        "",
    ],
)
def test_rejected_requirements(spec):
    with pytest.raises(SandboxError):
        executor.validate_requirement(spec)


def test_allowlist():
    allow = {"pandas", "numpy"}
    assert executor.validate_requirement("Pandas>=2", allow)
    with pytest.raises(SandboxError):
        executor.validate_requirement("requestz", allow)


def test_install_never_reaches_pip_for_bad_specs(monkeypatch):
    called = False

    async def fake_run(*a, **k):
        nonlocal called
        called = True

    monkeypatch.setattr(executor, "run_process", fake_run)
    with pytest.raises(SandboxError):
        asyncio.run(executor.install_package("--index-url=http://evil x", LIMITS))
    assert not called


def test_install_uses_wheels_only_and_sandbox_interpreter(monkeypatch):
    seen = {}

    async def fake_run(argv, **kwargs):
        seen["argv"], seen["env"] = argv, kwargs["env"]
        return executor.ExecResult(0, "", "")

    monkeypatch.setattr(executor, "run_process", fake_run)
    monkeypatch.setenv("SANDBOX_PYTHON", "/opt/sandbox-venv/bin/python")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-leak")
    asyncio.run(executor.install_package("pandas>=2", LIMITS))
    assert seen["argv"][0] == "/opt/sandbox-venv/bin/python"
    assert "--only-binary=:all:" in seen["argv"] and "uv" not in seen["argv"]
    assert "ANTHROPIC_API_KEY" not in seen["env"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses RLIMIT_NPROC")
def test_fork_bomb_is_contained(workdir):
    code = (
        "import os\n"
        "for _ in range(500):\n"
        "    try:\n"
        "        os.fork() or os._exit(0)\n"
        "    except OSError:\n"
        "        print('limited'); break"
    )
    result = run(workdir, code, SandboxLimits(timeout_s=10, max_processes=32))
    assert "limited" in result.stdout
