#!/usr/bin/env -S uv run --script --quiet
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# /// script
# dependencies = ["nox"]
# ///

"""Nox sessions."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import nox

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

nox.needs_version = ">=2025.10.16"
nox.options.default_venv_backend = "uv"


PYTHON_ALL_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]

_CAPPED_NUMERICAL_THREADS = {
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

_RELEASE_PACKAGE_SMOKE = """
import math
import os
from importlib.resources import files
from pathlib import Path

from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Result, Simulator, State

import mqt.yaqs as yaqs

package_file = Path(yaqs.__file__).resolve()
source_root = Path(os.environ["YAQS_SOURCE_ROOT"]).resolve()
if source_root in package_file.parents:
    raise RuntimeError(f"Imported YAQS from the source tree: {package_file}")
if not files("mqt.yaqs").joinpath("py.typed").is_file():
    raise RuntimeError("The built wheel does not contain mqt/yaqs/py.typed")

state = State(2, initial="zeros")
hamiltonian = Hamiltonian.ising(2, J=0.25, g=0.1)
params = AnalogSimParams(
    observables=[Observable("z", 0)],
    elapsed_time=0.1,
    dt=0.1,
    num_traj=1,
    sample_timesteps=False,
)
result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, params)
if not isinstance(result, Result):
    raise TypeError(f"Simulator.run returned {type(result).__name__}, not Result")
if len(result.expectation_values) != 1:
    raise RuntimeError("The installed-wheel analog smoke test returned the wrong number of observables")
expectation = complex(result.expectation_values[0][-1])
if not math.isfinite(expectation.real) or not math.isfinite(expectation.imag):
    raise RuntimeError("The installed-wheel analog smoke test returned an invalid expectation value")
"""

if os.environ.get("CI", None):
    nox.options.error_on_missing_interpreters = True


@contextlib.contextmanager
def preserve_lockfile() -> Generator[None]:
    """Preserve the lockfile by moving it to a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir_name:
        shutil.move("uv.lock", f"{temp_dir_name}/uv.lock")
        try:
            yield
        finally:
            shutil.move(f"{temp_dir_name}/uv.lock", "uv.lock")


@nox.session(reuse_venv=True, default=True)
def lint(session: nox.Session) -> None:
    """Run the linter."""
    if shutil.which("prek") is None:
        session.install("prek")

    session.run("prek", "run", "--all-files", *session.posargs, external=True)


def _run_tests(
    session: nox.Session,
    *,
    install_args: Sequence[str] = (),
    run_args: Sequence[str] = (),
    extra_torch: bool = False,
) -> None:
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}

    if "--cov" in session.posargs:
        # disable Numba JIT coverage
        env["NUMBA_DISABLE_JIT"] = "1"

    uv_args = [
        "uv",
        "run",
        "--no-dev",  # do not auto-install dev dependencies
        "--group",
        "test",
        *install_args,
    ]
    if extra_torch:
        uv_args.extend(["--extra", "torch"])

    session.run(
        *uv_args,
        "pytest",
        *run_args,
        *session.posargs,
        "--cov-config=pyproject.toml",
        env=env,
    )


@nox.session(python=PYTHON_ALL_VERSIONS, reuse_venv=True, default=True)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    _run_tests(session, run_args=["-m", "not release and not jit"], extra_torch=True)


@nox.session(python=PYTHON_ALL_VERSIONS, reuse_venv=True, venv_backend="uv")
def minimums(session: nox.Session) -> None:
    """Test the minimum versions of dependencies."""
    with preserve_lockfile():
        _run_tests(
            session,
            install_args=["--resolution=lowest-direct"],
            run_args=["-Wdefault", "-m", "not release and not jit"],
            extra_torch=True,
        )
        env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
        session.run("uv", "tree", "--frozen", env=env)


def _run_focused_tests(
    session: nox.Session,
    *,
    run_args: Sequence[str],
    env_overrides: Mapping[str, str],
) -> None:
    """Run a fixed serial test selection without optional test dependencies."""
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location, **env_overrides}
    session.run(
        "uv",
        "run",
        "--no-dev",
        "--with",
        "pytest>=9.0.1",
        "--with",
        "pytest-xdist[psutil]>=3.8",
        "pytest",
        "-o",
        "addopts=",
        "-ra",
        "--showlocals",
        "-p",
        "no:cacheprovider",
        *run_args,
        env=env,
    )


@nox.session(name="release-tests", python="3.14", reuse_venv=True)
def release_tests(session: nox.Session) -> None:
    """Run slow scientific validation without coverage or parallel pytest workers."""
    _run_focused_tests(
        session,
        run_args=["-m", "release", "tests/characterization/noise/optimization/test_run.py"],
        env_overrides=_CAPPED_NUMERICAL_THREADS,
    )


@nox.session(name="jit-tests", python="3.14", reuse_venv=True)
def jit_tests(session: nox.Session) -> None:
    """Compile and run focused Numba checks without coverage."""
    _run_focused_tests(
        session,
        run_args=[
            "-m",
            "jit",
            "tests/core/methods/test_lanczos_numba.py",
            "tests/core/methods/tdvp/test_numba.py",
        ],
        env_overrides={**_CAPPED_NUMERICAL_THREADS, "NUMBA_DISABLE_JIT": "0"},
    )


def _venv_python(venv_dir: Path) -> Path:
    """Return the Python executable for a virtual environment on this platform."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


@nox.session(name="release-package", python="3.14", reuse_venv=False)
def release_package(session: nox.Session) -> None:
    """Build and test the wheel from a clean temporary environment."""
    source_root = Path.cwd().resolve()
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        dist_dir = temp_dir / "dist"
        venv_dir = temp_dir / "venv"

        session.run("uv", "build", "--out-dir", dist_dir, external=True)
        wheels = list(dist_dir.glob("*.whl"))
        source_distributions = list(dist_dir.glob("*.tar.gz"))
        if len(wheels) != 1 or len(source_distributions) != 1:
            session.error(
                "Expected one wheel and one source distribution, "
                f"found {len(wheels)} wheel(s) and {len(source_distributions)} source distribution(s)."
            )

        session.run("uv", "venv", "--python", str(session.python), "--no-project", venv_dir, external=True)
        python = _venv_python(venv_dir)
        session.run("uv", "pip", "install", "--python", python, wheels[0], external=True)

        smoke_env = {
            **_CAPPED_NUMERICAL_THREADS,
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
            "YAQS_SOURCE_ROOT": str(source_root),
        }
        with session.chdir(temp_dir):
            session.run(
                python,
                "-W",
                "error",
                "-c",
                _RELEASE_PACKAGE_SMOKE,
                env=smoke_env,
                external=True,
            )


@nox.session(python="3.14", reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the docs. Use "--non-interactive" to avoid serving. Pass "-b linkcheck" to check links."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", dest="builder", default="html", help="Build target (default: html)")
    args, posargs = parser.parse_known_args(session.posargs)

    serve = args.builder == "html" and session.interactive
    if serve:
        session.install("sphinx-autobuild")

    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    shared_args = [
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        f"docs/_build/{args.builder}",
        *posargs,
    ]

    session.run(
        "uv",
        "run",
        "--no-dev",  # do not auto-install dev dependencies
        "--group",
        "docs",
        "sphinx-autobuild" if serve else "sphinx-build",
        *shared_args,
        env=env,
    )


if __name__ == "__main__":
    nox.main()
