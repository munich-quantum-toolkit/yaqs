#!/usr/bin/env python3
"""Validation for S_V^full and I_PT benchmark."""

from __future__ import annotations

import sys
from typing import cast

import numpy as np

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import DenseProcessTensor
from mqt.yaqs.characterization.memory.operational_memory.full_basis import enumerate_full_probe_catalog
from mqt.yaqs.memory_characterizer import make_zero_psi

TOL = 1e-10


def _pt(length: int, k: int, j: float, *, g: float = 1.0) -> DenseProcessTensor:
    return cast(
        DenseProcessTensor,
        MemoryCharacterizer(parallel=False, show_progress=False).build_process_tensor(
            Hamiltonian.ising(length=length, J=j, g=g),
            AnalogSimParams(dt=0.1, max_bond_dim=64, order=1),
            timesteps=[0.1] * (k + 1),
            return_type="dense",
            method="exhaustive",
            compress_every=1,
        ),
    )


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    for label, j, g, length in (
        ("A identity", 0.0, 0.0, 1),
        ("B unitary", 0.0, 1.0, 1),
        ("C markov g=0", 0.0, 0.0, 1),
    ):
        pt = _pt(length, 3, j, g=g)
        mi = pt.causal_block_mutual_information(2)
        check(f"{label} I_PT", float(mi["mutual_information"]) < TOL, f"I={mi['mutual_information']:.3e}")

    pt = _pt(6, 3, 1.0)
    mi = pt.causal_block_mutual_information(2)
    check("D interacting I_PT>0", float(mi["mutual_information"]) > 1e-8, f"I={mi['mutual_information']:.6f}")

    pt = _pt(6, 3, 0.0)
    mi = pt.causal_block_mutual_information(2)
    check("E J=0 I_PT", float(mi["mutual_information"]) < TOL, f"I={mi['mutual_information']:.3e}")

    catalog = enumerate_full_probe_catalog(cut=2, num_interventions=3)
    check("P_full size", len(catalog.past_settings) == 64, str(len(catalog.past_settings)))
    check("F_full size", len(catalog.future_settings) == 64, str(len(catalog.future_settings)))
    check("tomographic maps", catalog.intervention_tomographically_complete)
    check("output IC", catalog.output_informationally_complete)

    print("\n" + ("ALL TESTS PASSED" if not failures else f"FAILED: {failures}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
