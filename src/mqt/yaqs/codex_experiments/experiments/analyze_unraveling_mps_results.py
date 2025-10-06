from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@dataclass
class AnalysisSettings:
    precision_abs_tol: float = 1e-3
    precision_rel_tol: float = 0.05
    variance_rel_tol: float = 0.25


def _mean(values: np.ndarray | list[float] | tuple[float, ...]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _mean_qiskit_bonds(bonds: object) -> float:
    if isinstance(bonds, dict):
        per_layer = bonds.get("per_layer_mean_across_shots")
        if per_layer is not None:
            return _mean(per_layer)
        per_shot = bonds.get("per_shot_per_layer_max_bond_dim")
        if per_shot is not None:
            arr = np.asarray(per_shot, dtype=float)
            if arr.size == 0:
                return float("nan")
            return float(np.mean(arr))
        return float("nan")
    return _mean(bonds)


def _load_payload(path: Path) -> dict:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _iter_result_files(results_dir: Path) -> Iterable[Path]:
    for path in sorted(results_dir.glob("*.pkl")):
        if path.name == "analysis_summary.pkl":
            continue
        yield path


def analyze_file(path: Path, settings: AnalysisSettings) -> dict[str, object]:
    payload = _load_payload(path)

    mean_abs_errors = payload.get("mean_abs_errors", {})
    mae_standard = float(mean_abs_errors.get("standard", float("nan")))
    mae_qiskit = float(mean_abs_errors.get("qiskit_mps", float("nan")))
    precision_gap = mae_standard - mae_qiskit
    precision_close = bool(
        np.isfinite(mae_standard)
        and np.isfinite(mae_qiskit)
        and abs(precision_gap)
        <= max(settings.precision_abs_tol, settings.precision_rel_tol * max(mae_standard, mae_qiskit, 1e-12))
    )

    yaqs_bonds = payload.get("yaqs_bonds", {})
    mean_bond_standard = _mean(yaqs_bonds.get("standard", []))
    mean_bond_qiskit = _mean_qiskit_bonds(payload.get("qiskit_mps_bonds"))
    bond_advantage = mean_bond_qiskit - mean_bond_standard
    bond_lower = mean_bond_standard < mean_bond_qiskit

    variances = payload.get("stochastic_variances", {})
    mean_var_standard = _mean(variances.get("standard", []))
    mean_var_qiskit = _mean(variances.get("qiskit_mps", []))
    variance_gap = mean_var_qiskit - mean_var_standard
    variance_close = bool(
        np.isfinite(mean_var_standard)
        and np.isfinite(mean_var_qiskit)
        and abs(variance_gap)
        <= settings.variance_rel_tol * max(mean_var_standard, mean_var_qiskit, 1e-12)
    )

    preferred = bond_lower and precision_close and variance_close

    return {
        "file": path.name,
        "circuit_name": payload.get("circuit_name"),
        "num_qubits": payload.get("num_qubits"),
        "num_layers": payload.get("num_layers"),
        "noise_strength": payload.get("noise_strength"),
        "mae_standard": mae_standard,
        "mae_qiskit": mae_qiskit,
        "precision_gap": precision_gap,
        "precision_close": precision_close,
        "mean_bond_standard": mean_bond_standard,
        "mean_bond_qiskit": mean_bond_qiskit,
        "bond_advantage": bond_advantage,
        "bond_lower": bond_lower,
        "mean_var_standard": mean_var_standard,
        "mean_var_qiskit": mean_var_qiskit,
        "variance_gap": variance_gap,
        "variance_close": variance_close,
        "standard_preferred": preferred,
    }


def run_analysis(results_dir: Path, settings: AnalysisSettings) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for path in _iter_result_files(results_dir):
        try:
            summaries.append(analyze_file(path, settings))
        except Exception as exc:  # pragma: no cover - diagnostics
            print(f"Failed to analyze {path}: {exc}")
    return summaries


def _print_summary_table(summaries: list[dict[str, object]]) -> None:
    if not summaries:
        print("No result files found.")
        return

    header = (
        f"{'Circuit':>32}"
        f" {'Noise':>6}"
        f" {'MAE std':>9}"
        f" {'MAE mps':>9}"
        f" {'ΔMAE':>9}"
        f" {'⟨χ⟩ std':>9}"
        f" {'⟨χ⟩ mps':>9}"
        f" {'Δχ':>9}"
        f" {'⟨Var⟩ std':>11}"
        f" {'⟨Var⟩ mps':>11}"
        f" {'ΔVar':>9}"
        f" {'Preferred':>10}"
    )
    print(header)
    for summary in summaries:
        circuit = summary.get("circuit_name", "")
        noise = summary.get("noise_strength", float("nan"))
        row = (
            f"{circuit:>32}"
            f" {noise:6.3f}"
            f" {summary.get('mae_standard', float('nan')):9.3e}"
            f" {summary.get('mae_qiskit', float('nan')):9.3e}"
            f" {summary.get('precision_gap', float('nan')):9.3e}"
            f" {summary.get('mean_bond_standard', float('nan')):9.3f}"
            f" {summary.get('mean_bond_qiskit', float('nan')):9.3f}"
            f" {summary.get('bond_advantage', float('nan')):9.3f}"
            f" {summary.get('mean_var_standard', float('nan')):11.3e}"
            f" {summary.get('mean_var_qiskit', float('nan')):11.3e}"
            f" {summary.get('variance_gap', float('nan')):9.3e}"
            f" {('yes' if summary.get('standard_preferred') else 'no'):>10}"
        )
        print(row)

    preferred_cases = [row for row in summaries if row.get("standard_preferred")]
    print()
    if preferred_cases:
        print("Cases where YAQS (standard TJM) is preferred over Qiskit MPS:")
        for row in preferred_cases:
            print(
                f"  • {row['circuit_name']} (noise={row['noise_strength']:.3f})"
                f" — Δχ={row['bond_advantage']:.3f}, ΔMAE={row['precision_gap']:.2e}, ΔVar={row['variance_gap']:.2e}"
            )
    else:
        print("No configurations satisfied the preference criteria.")


def _write_summary_pickle(results_dir: Path, summaries: list[dict[str, object]]) -> Path:
    output_path = results_dir / "analysis_summary.pkl"
    serializable = [dict(item) for item in summaries]
    with output_path.open("wb") as fh:
        pickle.dump(serializable, fh)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YAQS vs Qiskit MPS benchmark results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing benchmark pickle files.",
    )
    parser.add_argument("--precision-abs-tol", type=float, default=1e-3)
    parser.add_argument("--precision-rel-tol", type=float, default=0.05)
    parser.add_argument("--variance-rel-tol", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = AnalysisSettings(
        precision_abs_tol=args.precision_abs_tol,
        precision_rel_tol=args.precision_rel_tol,
        variance_rel_tol=args.variance_rel_tol,
    )
    summaries = run_analysis(args.results_dir, settings)
    _print_summary_table(summaries)
    if summaries:
        output_path = _write_summary_pickle(args.results_dir, summaries)
        print(f"\nSaved tabulated summary to {output_path}")


if __name__ == "__main__":
    main()
