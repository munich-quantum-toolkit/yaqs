# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QASM loading utilities shared by the Simulator and EquivalenceChecker."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from qiskit import qasm2, qasm3
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.utils.optionals import HAS_QASM3_IMPORT

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qiskit.circuit import QuantumCircuit

_OPENQASM_VERSION_RE = re.compile(r"OPENQASM\s+(\d+)", re.IGNORECASE)
_QASM3_IMPORT_MSG = (
    "OpenQASM 3 loading requires the optional package qiskit-qasm3-import. "
    "Install with: pip install mqt-yaqs[qasm3] or pip install qiskit[qasm3-import]"
)
_INVALID_INPUT_MSG = (
    "Expected a QuantumCircuit, a filesystem path to an OpenQASM file, "
    "or a raw OpenQASM string whose first substantive line declares OPENQASM."
)


def _iter_substantive_lines(text: str) -> Iterator[str]:
    """Yield non-comment, non-blank lines from OpenQASM source text.

    Args:
        text: OpenQASM source text.

    Yields:
        Stripped substantive lines suitable for header and version detection.
    """
    in_block_comment = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
                remainder = stripped.split("*/", maxsplit=1)[1].strip()
                if not remainder or remainder.startswith("//"):
                    continue
                stripped = remainder
            else:
                continue

        while "/*" in stripped:
            before, _, after = stripped.partition("/*")
            if before.strip() and not before.strip().startswith("//"):
                stripped = before.strip()
                break
            if "*/" not in after:
                in_block_comment = True
                stripped = ""
                break
            stripped = after.split("*/", maxsplit=1)[1].strip()
        else:
            if not stripped or stripped.startswith("//"):
                continue

        yield stripped


def _parse_qasm_version(text: str) -> Literal["2", "3"] | None:
    """Parse the OpenQASM major version from source text.

    Args:
        text: OpenQASM source text.

    Returns:
        ``"2"`` or ``"3"`` when a version is recognized, otherwise ``None``.
    """
    for stripped in _iter_substantive_lines(text):
        if stripped.upper().startswith("OPENQASM"):
            match = _OPENQASM_VERSION_RE.match(stripped)
            if match is None:
                return "2"
            return "3" if match.group(1) == "3" else "2"
    return None


def _load_openqasm(text: str, *, path: Path | None = None) -> QuantumCircuit:
    """Load a QuantumCircuit from OpenQASM source text.

    Args:
        text: OpenQASM program text.
        path: Optional filesystem path for ``qasm*.load`` (preserves ``include`` resolution).

    Returns:
        The parsed ``QuantumCircuit``.

    Raises:
        ValueError: If ``text`` does not declare OpenQASM.
        ImportError: If OpenQASM 3 is requested but ``qiskit-qasm3-import`` is not installed.
    """
    version = _parse_qasm_version(text)
    if version is None:
        raise ValueError(_INVALID_INPUT_MSG)

    if version == "3":
        if not HAS_QASM3_IMPORT:
            raise ImportError(_QASM3_IMPORT_MSG)
        try:
            if path is not None:
                return qasm3.load(str(path))
            loads_text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
            loads_text = "\n".join(line.split("//", maxsplit=1)[0] for line in loads_text.splitlines())
            return qasm3.loads(loads_text)
        except MissingOptionalLibraryError as exc:
            raise ImportError(_QASM3_IMPORT_MSG) from exc

    if path is not None:
        return qasm2.load(str(path))
    loads_text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    loads_text = "\n".join(line.split("//", maxsplit=1)[0] for line in loads_text.splitlines())
    return qasm2.loads(loads_text)


def load_circuit(circuit: QuantumCircuit | str | Path) -> QuantumCircuit:
    """Load a QuantumCircuit from OpenQASM or return it unchanged.

    Accepts a ``QuantumCircuit``, a ``Path`` to an OpenQASM file, or a ``str`` that is
    either raw OpenQASM source (when the first substantive line declares ``OPENQASM``)
    or a filesystem path. Prefer file paths when the program uses ``include`` directives.

    OpenQASM 3 requires the optional package ``qiskit-qasm3-import``
    (``pip install mqt-yaqs[qasm3]``).

    Args:
        circuit: Circuit object, OpenQASM path, or OpenQASM source string.

    Returns:
        The corresponding ``QuantumCircuit``.

    Raises:
        ValueError: If ``circuit`` is not a recognized input form.
    """
    if not isinstance(circuit, (str, Path)):
        return circuit

    if isinstance(circuit, str) and _parse_qasm_version(circuit) is not None:
        return _load_openqasm(circuit)

    path = Path(circuit)
    if not path.is_file():
        raise ValueError(_INVALID_INPUT_MSG)

    return _load_openqasm(path.read_text(encoding="utf-8"), path=path)
