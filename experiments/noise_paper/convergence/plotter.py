from __future__ import annotations

import re
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Helpers: load + reconstruct
# ----------------------------
_SUFFIX_RE = re.compile(r".*_(\d+)\.pickle$")


def _extract_suffix_int(path: Path) -> int:
    m = _SUFFIX_RE.match(path.name)
    if not m:
        raise ValueError(f"Could not parse numeric suffix from: {path.name}")
    return int(m.group(1))


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def reconstruct_series_from_snapshots(paths: list[Path]) -> list[Any]:
    """
    Reconstruct a full list of per-dp entries from a set of pickle snapshots.

    Works for BOTH common conventions:
      (A) CUMULATIVE snapshot: file_k contains results for dp indices [0..k-1]
          (this is what your loop currently writes)
      (B) CHUNKED snapshot: file_k contains ONLY the last chunk (e.g., 10 dp's)

    Strategy:
      - sort by suffix
      - append only the "new" tail when file contents are cumulative
      - otherwise append whole file contents
    """
    if not paths:
        return []

    paths = sorted(paths, key=_extract_suffix_int)

    out: list[Any] = []
    prev_len = 0

    for p in paths:
        data = load_pickle(p)

        if not isinstance(data, list):
            raise TypeError(f"{p.name} did not contain a list; got {type(data)}")

        # Case A: cumulative snapshot (length grows)
        if len(data) >= prev_len and data[:prev_len] == out[:prev_len]:
            out.extend(data[prev_len:])
            prev_len = len(data)
        else:
            # Case B: chunked snapshot (just append)
            out.extend(data)
            prev_len = len(out)

    return out


# ----------------------------
# Extract time series
# ----------------------------
def extract_yaqs_xx_timeseries(dp_entry: Any, obs_index: int = 0) -> np.ndarray:
    """
    dp_entry is what you appended as `cost` in results1/results2:
        cost == sim_params.observables  (a list of Observable objects)

    We take cost[0].results as the XX trajectory.
    """
    # dp_entry: list[Observable]
    obs_list = dp_entry
    if not isinstance(obs_list, (list, tuple)) or len(obs_list) <= obs_index:
        raise ValueError(f"Unexpected YAQS entry format: {type(dp_entry)}")

    obs = obs_list[obs_index]
    if not hasattr(obs, "results"):
        raise ValueError("YAQS Observable object has no attribute `results`")

    ts = np.asarray(obs.results, dtype=float)
    return ts


def extract_qutip_timeseries(dp_entry: Any) -> np.ndarray:
    """
    dp_entry is what you appended as `cost` in results3:
        cost == Z_results (a python list of floats)
    """
    return np.asarray(dp_entry, dtype=float)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    d = a[:n] - b[:n]
    return float(np.mean(d * d))


# ----------------------------
# Main analysis
# ----------------------------
def main(
    data_dir: str | Path = ".",
    dt: float = 0.1,
    dp_min: float = 1e-4,
    dp_max: float = 1.0,
    num_dp: int = 100,
) -> None:
    data_dir = Path(data_dir)

    # Rebuild dp list exactly as in your generator script
    dp_list = np.logspace(np.log10(dp_min), np.log10(dp_max), num_dp)

    u1_paths = sorted(data_dir.glob("convergence_u1_*.pickle"), key=_extract_suffix_int)
    u2_paths = sorted(data_dir.glob("convergence_u2_*.pickle"), key=_extract_suffix_int)
    qt_paths = sorted(data_dir.glob("convergence_qutip_*.pickle"), key=_extract_suffix_int)

    if not (u1_paths and u2_paths and qt_paths):
        raise FileNotFoundError(
            f"Missing pickle files. Found: u1={len(u1_paths)}, u2={len(u2_paths)}, qutip={len(qt_paths)} "
            f"in {data_dir.resolve()}"
        )

    results_u1 = reconstruct_series_from_snapshots(u1_paths)
    results_u2 = reconstruct_series_from_snapshots(u2_paths)
    results_qt = reconstruct_series_from_snapshots(qt_paths)

    # Align lengths (in case the run was interrupted mid-sweep)
    n = min(len(results_u1), len(results_u2), len(results_qt), len(dp_list))
    results_u1 = results_u1[:n]
    results_u2 = results_u2[:n]
    results_qt = results_qt[:n]
    dp_list = dp_list[:n]

    mse_u1 = np.empty(n, dtype=float)
    mse_u2 = np.empty(n, dtype=float)

    for i in range(n):
        yaqs_u1_ts = extract_yaqs_xx_timeseries(results_u1[i], obs_index=0)
        yaqs_u2_ts = extract_yaqs_xx_timeseries(results_u2[i], obs_index=0)
        qutip_ts = extract_qutip_timeseries(results_qt[i])

        mse_u1[i] = mse(yaqs_u1_ts, qutip_ts)
        mse_u2[i] = mse(yaqs_u2_ts, qutip_ts)

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure()
    plt.loglog(dp_list, mse_u1, marker="o", markersize=3, linewidth=1, label="YAQS u1 vs QuTiP")
    plt.loglog(dp_list, mse_u2, marker="o", markersize=3, linewidth=1, label="YAQS u2 vs QuTiP")
    plt.xlabel("dp")
    plt.ylabel("MSE over full trajectory (t in [0, T])")
    plt.title(f"MSE vs dp (dt={dt}, N={n})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: save numeric data
    out = np.column_stack([dp_list, mse_u1, mse_u2])
    np.savetxt(data_dir / "mse_vs_dp.csv", out, delimiter=",", header="dp,mse_u1,mse_u2", comments="")
    print(f"Saved: {data_dir / 'mse_vs_dp.csv'}")


if __name__ == "__main__":
    main(
        data_dir=".",   # folder containing the pickles
        dt=0.1,
        dp_min=1e-4,
        dp_max=1.0,
        num_dp=100,
    )
