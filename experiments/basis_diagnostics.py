"""Print conditioning diagnostics for tomography bases.

Usage:
  python -m experiments.basis_diagnostics --basis_seed 12345
"""

from __future__ import annotations

import argparse

from mqt.yaqs.characterization.tomography.basis import (
    calculate_dual_choi_basis,
    dual_norm_metrics,
    get_choi_basis,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Tomography basis conditioning diagnostics.")
    p.add_argument("--basis_seed", type=int, default=12345, help="Seed for basis='random'.")
    args = p.parse_args()

    bases = [
        ("standard", None),
        ("tetrahedral", None),
        ("random", args.basis_seed),
    ]

    for name, seed in bases:
        choi_basis, _ = get_choi_basis(basis=name, seed=seed)
        duals = calculate_dual_choi_basis(choi_basis)
        dn = dual_norm_metrics(duals)
        print(f"basis={name} seed={seed}")
        print(f"  mean_dual_norm={dn['mean_dual_norm']:.6g}")
        print(f"  max_dual_norm={dn['max_dual_norm']:.6g}")
        print("")


if __name__ == "__main__":
    main()
