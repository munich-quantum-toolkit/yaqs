"""Shared helpers for V-matrix paper benchmarks on :class:`~mqt.yaqs.MemoryCharacterizer`."""

from __future__ import annotations

import copy
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.exact import ExactBackend
from mqt.yaqs.characterization.memory.operational_memory.response_matrix import (
    assemble_response_matrix,
    compute_spectrum,
)
from mqt.yaqs.characterization.memory.operational_memory.results import (
    CharacterizationResult,
    pack_result,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import (
    ProbeSet,
    sample_cut_measurement,
    sample_cut_preparation,
    sample_probe,
    sample_probes,
)
from mqt.yaqs.memory_characterizer import make_zero_psi

# Paper defaults (L=6 Ising chain unless a benchmark sweeps L).
L_DEFAULT = 6
K_DEFAULT = 20
DT_DEFAULT = 0.1
G_DEFAULT = 1.0
J_SWEEP = [0.05 * i for i in range(41)]
BETA = 1.0
HEATMAP_VMIN = 1e-3
HEATMAP_VMAX = 3.0
PANEL2_CUTS: tuple[int, ...] = (10, 15, 19)
PANEL3_JS: tuple[float, ...] = (0.4, 1.0, 2.0)

# Back-compat aliases for any stale imports.
L_FIXED = L_DEFAULT
K_FIXED = K_DEFAULT
DT_FIXED = DT_DEFAULT
G_FIXED = G_DEFAULT
J_SWEEP_DEFAULT = J_SWEEP
BRANCH_WEIGHT_BETA = BETA
HEATMAP_COLOR_VMIN = HEATMAP_VMIN
HEATMAP_COLOR_VMAX = HEATMAP_VMAX
PANEL2_FIXED_CUTS = PANEL2_CUTS
PANEL3_TARGET_JS = PANEL3_JS


def parse_int_list(spec: str) -> list[int]:
    """Parse a comma-separated list of integers."""
    vals = [int(tok.strip()) for tok in spec.split(",") if tok.strip()]
    if not vals:
        raise ValueError("expected at least one integer")
    return vals


def parse_float_list(spec: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    vals = [float(tok.strip()) for tok in spec.split(",") if tok.strip()]
    if not vals:
        raise ValueError("expected at least one float")
    return vals


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write benchmark rows to CSV."""
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load benchmark CSV rows."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


write_summary_csv = write_csv
load_summary_csv = load_csv


def random_qubit_state(rng: np.random.Generator) -> np.ndarray:
    """Sample a normalized random qubit pure state."""
    psi = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    psi = psi.astype(np.complex128)
    return psi / max(float(np.linalg.norm(psi)), 1e-15)


def initial_states_sys_env0(*, length: int, n_seeds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Site 0 random (or |0> for one seed); remaining sites fixed to |0>."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")
    if n_seeds == 1:
        return [make_zero_psi(length)]
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    out: list[np.ndarray] = []
    for _ in range(n_seeds):
        psi = random_qubit_state(rng)
        for _ in range(length - 1):
            psi = np.kron(psi, z)
        out.append(psi / max(float(np.linalg.norm(psi)), 1e-15))
    return out


list_initial_states_sys_env0 = initial_states_sys_env0
random_pure_state = random_qubit_state


def ising_chain(*, length: int, j: float, g: float = G_DEFAULT) -> Hamiltonian:
    """Build an encoded Ising Hamiltonian."""
    ham = Hamiltonian.ising(length=length, J=float(j), g=float(g))
    ham.ensure_encoded("mpo")
    return ham


make_ising_hamiltonian = ising_chain


def sim_params(*, dt: float = DT_DEFAULT) -> AnalogSimParams:
    """MCWF-oriented params matching legacy experiment defaults."""
    return AnalogSimParams(dt=float(dt))


default_sim_params = sim_params


def characterizer(*, parallel: bool = True) -> MemoryCharacterizer:
    """Configured :class:`MemoryCharacterizer` for exact benchmarks."""
    return MemoryCharacterizer(parallel=parallel, show_progress=False)


def characterize(
    mc: MemoryCharacterizer,
    ham: Hamiltonian,
    params: AnalogSimParams,
    *,
    k: int,
    cut: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator | None = None,
    probe_set: ProbeSet | CharacterizationResult | None = None,
    initial_psi: np.ndarray | None = None,
    style: str = "haar",
    center: bool = True,
) -> CharacterizationResult:
    """Run split-cut characterization via the public API."""
    if center:
        return mc.characterize(
            ham,
            params,
            num_interventions=int(k),
            cut=int(cut),
            n_pasts=int(n_pasts),
            n_futures=int(n_futures),
            rng=rng,
            probe_set=probe_set,
            initial_psi=initial_psi,
            intervention_style=style,
        )

    if isinstance(probe_set, CharacterizationResult):
        probe_set = probe_set.probes(int(cut)).get("probe_set")
    if probe_set is None:
        probe_set = sample_probes(
            cut=int(cut),
            num_interventions=int(k),
            n_pasts=int(n_pasts),
            n_futures=int(n_futures),
            rng=rng,
            intervention_style=style,
        )
    psi0 = np.asarray(initial_psi if initial_psi is not None else make_zero_psi(ham.length), dtype=np.complex128)
    ham.ensure_encoded("mpo")
    backend = ExactBackend(
        operator=ham.mpo,
        sim_params=params,
        initial_psi=psi0,
        parallel=mc.parallel,
        show_progress=False,
        solver="MCWF",
    )
    pauli_xyz_ij, weights_ij = backend.evaluate_probes_weighted(probe_set)
    _raw, response_matrix = assemble_response_matrix(
        pauli_xyz_ij,
        weights_ij,
        beta=BETA,
        center=False,
        log_weight_warnings=False,
    )
    ana = compute_spectrum(response_matrix, discarded_weight_threshold=None)
    out: dict[str, Any] = {
        **ana,
        "response_matrix": response_matrix,
        "probe_set": probe_set,
        "weights_ij": weights_ij,
        "pauli_xyz_ij": pauli_xyz_ij,
    }
    return pack_result(out, cut=int(cut))


def metrics_from(result: CharacterizationResult, cut: int) -> dict[str, float | int]:
    """Extract scalar diagnostics used in benchmark CSVs."""
    sv = result.singular_values(cut)
    rank = int(np.sum(np.asarray(sv, dtype=np.float64) > 1e-12))
    return {
        "entropy": float(result.entropy(cut)),
        "rank": rank,
        "delta_norm": float(np.linalg.norm(result.response_matrix(cut))),
    }


def mean_metrics(results: list[dict[str, float | int]]) -> dict[str, float | int]:
    """Average entropy / rank / delta_norm across initial states."""
    ent = [float(r["entropy"]) for r in results]
    rank = [int(r["rank"]) for r in results]
    delta = [float(r["delta_norm"]) for r in results]
    return {
        "entropy": float(np.mean(ent)),
        "entropy_std": float(np.std(ent, ddof=1)) if len(ent) > 1 else 0.0,
        "rank": int(round(float(np.mean(rank)))),
        "delta_norm": float(np.mean(delta)),
    }


def sample_cut_probes(
    *,
    cut: int,
    k: int,
    n_pasts: int,
    n_futures: int,
    seed: int,
    style: str = "haar",
) -> ProbeSet:
    """Sample one probe grid; RNG depends only on ``seed`` and ``cut``."""
    rng = np.random.default_rng(int(seed) + 10_000 * int(cut))
    return sample_probes(
        cut=int(cut),
        num_interventions=int(k),
        n_pasts=int(n_pasts),
        n_futures=int(n_futures),
        rng=rng,
        intervention_style=style,
    )


def characterize_custom_sequences(
    mc: MemoryCharacterizer,
    ham: Hamiltonian,
    params: AnalogSimParams,
    *,
    probe_set: ProbeSet,
    psi_pairs_list: list[list[Any]],
    initial_psi: np.ndarray,
    cut: int,
) -> CharacterizationResult:
    """Characterize a custom intervention geometry (e.g. ell-delay bridge).

    Uses the exact backend directly because ``characterize()`` does not yet accept
    pre-built ``psi_pairs_list`` grids.
    """
    ham.ensure_encoded("mpo")
    backend = ExactBackend(
        operator=ham.mpo,
        sim_params=params,
        initial_psi=np.asarray(initial_psi, dtype=np.complex128),
        parallel=mc.parallel,
        show_progress=False,
        solver="MCWF",
    )
    pauli_xyz_ij, weights_ij = backend.evaluate_probes_weighted(
        probe_set,
        psi_pairs_list=psi_pairs_list,
    )
    _raw, response_matrix = assemble_response_matrix(
        pauli_xyz_ij,
        weights_ij,
        beta=BETA,
        center=True,
        log_weight_warnings=False,
    )
    ana = compute_spectrum(response_matrix, discarded_weight_threshold=None)
    out: dict[str, Any] = {
        **ana,
        "response_matrix": response_matrix,
        "probe_set": probe_set,
        "weights_ij": weights_ij,
        "pauli_xyz_ij": pauli_xyz_ij,
    }
    return pack_result(out, cut=int(cut))


def evaluate_weighted_probes(
    mc: MemoryCharacterizer,
    ham: Hamiltonian,
    params: AnalogSimParams,
    *,
    probe_set: ProbeSet,
    initial_psi: np.ndarray,
    psi_pairs_list: list[list[Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate full weighted probe responses (for convergence prefix slicing)."""
    ham.ensure_encoded("mpo")
    backend = ExactBackend(
        operator=ham.mpo,
        sim_params=params,
        initial_psi=np.asarray(initial_psi, dtype=np.complex128),
        parallel=mc.parallel,
        show_progress=False,
        solver="MCWF",
    )
    return backend.evaluate_probes_weighted(probe_set, psi_pairs_list=psi_pairs_list)


def slice_probe_prefix(
    pauli_xyz_ij: np.ndarray,
    weights_ij: np.ndarray,
    m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice evaluated probe responses to an ``m x m`` prefix grid."""
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    p = np.asarray(pauli_xyz_ij)
    if p.shape[0] < m or p.shape[1] < m:
        raise ValueError(f"cannot slice pauli_xyz_ij shape={p.shape} to m={m}")
    p_sub = p[:m, :m, ...]
    w = np.asarray(weights_ij)
    if w.ndim >= 2 and w.shape[0] >= m and w.shape[1] >= m:
        w_sub = w[:m, :m, ...]
    elif w.shape[0] >= m:
        w_sub = w[:m, ...]
    else:
        raise ValueError(f"cannot slice weights_ij shape={w.shape} to m={m}")
    return np.asarray(p_sub), np.asarray(w_sub)


def entropy_from_responses(pauli_xyz_sub: np.ndarray, weights_sub: np.ndarray) -> float:
    """Compute :math:`S_V` from sliced Pauli responses and branch weights."""
    _raw, response_matrix = assemble_response_matrix(
        pauli_xyz_sub,
        weights_sub,
        beta=BETA,
        center=True,
        log_weight_warnings=False,
    )
    ana = compute_spectrum(memory_matrix, discarded_weight_threshold=None)
    return float(ana["entropy"])


entropy_from_pauli_weights = entropy_from_responses


def singular_value_probs(s: np.ndarray) -> np.ndarray:
    """Normalized squared singular values."""
    p = np.asarray(s, dtype=np.float64) ** 2
    ps = float(np.sum(p))
    if ps <= 0.0:
        return np.zeros(0, dtype=np.float64)
    return p / ps


def sample_ell_base_ensemble(
    *,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    past_len: int,
    future_len: int,
    style: str = "haar",
) -> tuple[list[list[Any]], list[np.ndarray], list[np.ndarray], list[list[Any]]]:
    """Sample one past/future probe ensemble for ell-delay sweeps."""
    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for _ in range(n_pasts):
        pairs_i = [sample_probe(rng, intervention_style=style)[1] for _ in range(past_len)]
        _feat_m, psi_m = sample_cut_measurement(rng)
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for _ in range(n_futures):
        _feat_p, psi_p = sample_cut_preparation(rng)
        future_prep_cut.append(psi_p)
        future_pairs.append([sample_probe(rng, intervention_style=style)[1] for _ in range(future_len)])
    return past_pairs, past_cut_meas, future_prep_cut, future_pairs


def build_ell_delay_probes(
    *,
    past_pairs: list[list[Any]],
    past_cut_meas: list[np.ndarray],
    future_prep_cut: list[np.ndarray],
    future_pairs: list[list[Any]],
    past_len: int,
    future_len: int,
    ell: int,
) -> tuple[ProbeSet, list[list[Any]]]:
    """Assemble delayed-bridge probes from a fixed past/future ensemble."""
    n_pasts = len(past_pairs)
    n_futures = len(future_pairs)
    left_cut = int(past_len + 1)
    ell_i = int(ell)
    k_this = int(past_len + 1 + ell_i + 1 + future_len)
    z0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    all_pairs: list[list[Any]] = []
    for i in range(n_pasts):
        for j in range(n_futures):
            full: list[Any] = list(past_pairs[i])
            full.append((np.asarray(past_cut_meas[i], dtype=np.complex128), z0))
            for _ in range(ell_i):
                full.append((z0, z0))
            full.append((z0, np.asarray(future_prep_cut[j], dtype=np.complex128)))
            full.extend(future_pairs[j])
            all_pairs.append(full)

    probe_set = ProbeSet(
        cut=left_cut,
        num_interventions=k_this,
        past_features=np.zeros((n_pasts, max(1, past_len + 1), 32), dtype=np.float32),
        future_features=np.zeros((n_futures, max(1, 1 + ell_i + future_len), 32), dtype=np.float32),
        past_pairs=copy.deepcopy(past_pairs),
        past_cut_meas=[np.asarray(x, dtype=np.complex128) for x in past_cut_meas],
        future_prep_cut=[np.asarray(x, dtype=np.complex128) for x in future_prep_cut],
        future_pairs=copy.deepcopy(future_pairs),
    )
    return probe_set, all_pairs


def build_ell_delay_probes_fresh(
    *,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    past_len: int,
    future_len: int,
    ell: int,
    style: str = "haar",
) -> tuple[ProbeSet, list[list[Any]]]:
    """Sample a fresh past/future ensemble and build ell-delay probes."""
    base = sample_ell_base_ensemble(
        n_pasts=n_pasts,
        n_futures=n_futures,
        rng=rng,
        past_len=past_len,
        future_len=future_len,
        style=style,
    )
    return build_ell_delay_probes(
        past_pairs=base[0],
        past_cut_meas=base[1],
        future_prep_cut=base[2],
        future_pairs=base[3],
        past_len=past_len,
        future_len=future_len,
        ell=ell,
    )


def configure_matplotlib_prl() -> None:
    """Physical Review Letters–oriented matplotlib defaults."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "grid.alpha": 0.0,
            "lines.linewidth": 1.5,
            "lines.markersize": 3.5,
        }
    )


configure_matplotlib_prl_figure = configure_matplotlib_prl


def configure_matplotlib() -> None:
    """Lightweight matplotlib defaults for quick line plots."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "font.family": "sans-serif",
            "mathtext.default": "it",
        }
    )


def save_figure(fig: object, path_stem: Path) -> None:
    """Save a figure as PDF and PNG."""
    import matplotlib.pyplot as plt

    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


savefig_base = save_figure
