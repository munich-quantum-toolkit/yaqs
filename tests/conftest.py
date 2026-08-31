# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared pytest configuration and constants for YAQS tests."""

from __future__ import annotations

import os
from importlib.util import find_spec
from typing import TYPE_CHECKING

import psutil
import pytest

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path


def set_default_thread_limits(environment: MutableMapping[str, str]) -> None:
    """Default numerical thread pools to one thread.

    Args:
        environment: Environment mapping to configure.
    """
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        environment.setdefault(name, "1")


# Set defaults before test modules import numerical libraries. Explicit values
# from a local shell or CI runner take precedence.
set_default_thread_limits(os.environ)

# Default seed for stochastic integration tests (TJM, noisy simulator.run, etc.).
YAQS_TEST_SEED = 42


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:
    """Select an allocation-aware pytest-xdist worker count.

    Returns:
        The number of xdist workers.
    """
    del config
    # Delay the YAQS import until numerical thread defaults are configured.
    from mqt.yaqs.core.parallel_utils import available_cpus  # ruff: ignore[import-outside-top-level]

    worker_limit = available_cpus()
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return worker_limit

    physical_cpus = psutil.cpu_count(logical=False) or worker_limit
    available_physical_cpus = min(physical_cpus, worker_limit)
    return max(1, available_physical_cpus - 1)


LARGE_QASM2_STRING = """\
// circuit_small.qasm
OPENQASM 2.0;
include "qelib1.inc";

// teach the parser what rzz is
gate rzz(theta) a,b {
    cx a,b;
    rz(theta) b;
    cx a,b;
}

qreg q[6];
creg c[6];

// layer 1
ry(0.671272670484062) q[0];
ry(0.383393090311768) q[1];
ry(0.390761880931935) q[2];
// entangle first three
cx q[0],q[1];
cx q[1],q[2];

// layer 2
ry(0.105246328897056) q[3];
cx q[2],q[3];

// layer 3
ry(0.738872295810560) q[4];
cx q[3],q[4];

// layer 4
ry(0.277312276421170) q[5];
cx q[4],q[5];

// now use rzz
rzz(1.5707963267948966) q[0],q[5];

// fan-in back to q0
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];

// final mixing
ry(0.55) q[0];

// seal it
barrier q;
"""

SAMPLE_QASM3_STRING = """\
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];
c = measure q;
"""


def write_qasm_file(directory: Path, content: str, *, filename: str = "circuit.qasm") -> Path:
    """Write OpenQASM source to a temporary file for path-based loading tests.

    Args:
        directory: Directory in which to create the file.
        content: OpenQASM source text to write.
        filename: Name of the file to create within ``directory``.

    Returns:
        Path to the written OpenQASM file.
    """
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path


requires_qasm3_import = pytest.mark.skipif(
    find_spec("qiskit_qasm3_import") is None,
    reason="qiskit-qasm3-import is not installed",
)
