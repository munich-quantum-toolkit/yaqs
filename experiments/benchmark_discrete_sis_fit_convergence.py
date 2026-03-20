"""Discrete SIS (:func:`_estimate_sis`) + :meth:`DenseComb.fit` vs exhaustive Υ.

After each run of :func:`mqt.yaqs.characterization.tomography.process_tomography._estimate_sis`
(sequential importance sampling over intervention indices), fits a dense comb
with :meth:`DenseComb.fit`, which minimizes

    Σ_{α: w_α>0} ‖ ρ_pred(α; Υ) − w_α ρ_out(α) ‖_F² + λ ‖Υ‖_F²

using the same forward map as ``predict_from_upsilon`` (primal ``choi_basis``).
Records Frobenius error of Υ vs exhaustive reference and state-prediction error
vs the exhaustive comb.

**Post-hoc low-rank diagnostic:** the fitted Υ is truncated with an SVD (best
rank-r approximation in Frobenius norm) for several ranks ``r``, and the same
prediction errors are recomputed. This probes whether the learned comb is
approximately low-rank *after* fitting, without a nonconvex low-rank constraint
during optimization. (A more flexible direct low-rank parameterization later
would be Υ = A B† rather than X X†.)

Use the **same** ``basis`` / ``basis_seed`` for exhaustive and SIS.

Outputs:
  - per-seed raw CSV
  - mean/std summary CSV
  - log–log slope vs N (excluding N = 16^k)

Usage:
  python -m experiments.benchmark_discrete_sis_fit_convergence --L 2 --ks 2 --n_seeds 5
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
import itertools
from pathlib import Path

import numpy as np

from mqt.yaqs.characterization.tomography.combs import DenseComb
from mqt.yaqs.characterization.tomography.metrics import rel_fro_error
from mqt.yaqs.characterization.tomography.process_tomography import (
    _estimate_sis,
    _sequence_data_to_dense,
    run,
)
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def _predict_from_upsilon_primal(
    upsilon: np.ndarray,
    alphas: tuple[int, ...],
    choi_basis: list,
) -> np.ndarray:
    """Same contraction as ``TomographyEstimate.reconstruct_comb_choi`` / ``DenseComb.fit``."""
    past = choi_basis[alphas[0]]
    for a in alphas[1:]:
        past = np.kron(past, choi_basis[a])
    k = len(alphas)
    dim_p = 4**k
    u4 = upsilon.reshape(2, dim_p, 2, dim_p)
    ins = past.T.reshape(dim_p, dim_p)
    return np.einsum("s p q r, r p -> s q", u4, ins)


def _svd_truncate_frobenius(U: np.ndarray, rank: int) -> np.ndarray:
    """Best rank-``rank`` approximation to ``U`` in Frobenius norm (truncated SVD)."""
    r = int(rank)
    if r <= 0:
        return np.zeros_like(U)
    u, s, vh = np.linalg.svd(U, full_matrices=False)
    r_eff = min(r, int(s.shape[0]))
    return (u[:, :r_eff] * s[:r_eff]) @ vh[:r_eff, :]


def _constraint_ls_mean_sq_residual(
    upsilon: np.ndarray,
    estimate,
    *,
    weight_tol: float,
) -> tuple[float, int]:
    """Mean squared Frobenius residual matching :meth:`DenseComb.fit` targets."""
    k = int(estimate.tensor.ndim - 1)
    basis = estimate.choi_basis
    if basis is None:
        return float("nan"), 0
    tot = 0.0
    n = 0
    for alphas in itertools.product(range(16), repeat=k):
        w = float(estimate.weights[alphas])
        if w <= weight_tol:
            continue
        rho_out = estimate.tensor[(slice(None), *alphas)].reshape(2, 2)
        rho_p = _predict_from_upsilon_primal(upsilon, alphas, basis)
        diff = rho_p - w * rho_out
        tot += float(np.real(np.sum(diff.conj() * diff)))
        n += 1
    if n == 0:
        return float("nan"), 0
    return tot / float(n), n


@dataclass
class Config:
    ks: list[int] = field(default_factory=lambda: [2])
    L: int = 2
    dt: float = 0.1
    n_seeds: int = 10
    seed_base: int = 12345
    out_dir: Path = Path("experiments_output/discrete_sis_fit_convergence")
    prep_mixture_eps: float = 0.1
    basis: str = "tetrahedral"
    basis_seed: int | None = 12345
    proposal: str = "local"
    solver: str = "MCWF"
    max_bond_dim: int = 16
    fit_lam: float = 1e-8
    fit_weight_tol: float = 1e-30
    svd_trunc_ranks: list[int] = field(default_factory=lambda: [4, 8, 16, 32])


def _default_Ns(k: int) -> list[int]:
    n_exact = 16**k
    Ns: list[int] = []
    n = 170
    while n < n_exact:
        Ns.append(n)
        n += 2
    Ns.append(n_exact)
    return sorted(set(int(x) for x in Ns))


def _fit_loglog_slope(Ns: np.ndarray, errs: np.ndarray) -> float:
    mask = (Ns > 0) & np.isfinite(errs) & (errs > 0)
    if int(np.sum(mask)) < 2:
        return float("nan")
    x = np.log(Ns[mask].astype(float))
    y = np.log(errs[mask].astype(float))
    a, _b = np.polyfit(x, y, deg=1)
    return float(a)


def run_benchmark(cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cfg.out_dir / "sis_fit_convergence_raw.csv"
    summary_path = cfg.out_dir / "sis_fit_convergence_summary.csv"

    op = MPO.ising(length=cfg.L, J=0.0, g=0.1)
    params = AnalogSimParams(
        dt=cfg.dt,
        solver=cfg.solver,
        show_progress=False,
        max_bond_dim=cfg.max_bond_dim,
    )

    variants = [
        {"name": f"sis_{cfg.proposal}_fit_ls", "proposal": cfg.proposal},
    ]

    trunc_frob_cols = [f"err_trunc_r{r}_pred_state_frob" for r in cfg.svd_trunc_ranks]
    trunc_trace_cols = [f"err_trunc_r{r}_pred_state_trace" for r in cfg.svd_trunc_ranks]

    def _random_interventions(rng: np.random.Generator, k: int) -> list:
        interventions: list = []
        for _ in range(k):
            h = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
            u, _ = np.linalg.qr(h)

            def ch(rho: np.ndarray, u_sub=u) -> np.ndarray:
                return u_sub @ rho @ u_sub.conj().T

            interventions.append(ch)
        return interventions

    def _trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
        return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(rho - sigma))))

    raw_columns = [
        "variant",
        "k",
        "N",
        "seed",
        "n_sequences",
        "n_observed_cells",
        "fit_lam",
        "fit_weight_tol",
        "mean_sq_residual_constraints",
        "err_fit_U_vs_exhaustive",
        "err_fit_pred_state_frob",
        "err_fit_pred_state_trace",
        "prefactor_fit_pred",
        *trunc_frob_cols,
        *trunc_trace_cols,
    ]

    float_keys = [
        "n_sequences",
        "n_observed_cells",
        "mean_sq_residual_constraints",
        "err_fit_U_vs_exhaustive",
        "err_fit_pred_state_frob",
        "err_fit_pred_state_trace",
        "prefactor_fit_pred",
        *trunc_frob_cols,
        *trunc_trace_cols,
    ]

    with raw_path.open("w", newline="") as f_raw:
        w = csv.writer(f_raw)
        w.writerow(raw_columns)

        rng_global = np.random.default_rng(cfg.seed_base)

        for k in cfg.ks:
            timesteps = [cfg.dt] * k
            Ns = _default_Ns(k)

            comb_ex = run(
                op,
                params,
                timesteps=timesteps,
                method="exhaustive",
                output="dense",
                parallel=False,
                num_trajectories=1,
                basis=cfg.basis,
                basis_seed=cfg.basis_seed,
            )
            U_ex = comb_ex.to_matrix()

            intervention_sets = [
                _random_interventions(rng_global, k) for _ in range(3)
            ]

            for variant in variants:
                for N in Ns:
                    for s in range(cfg.n_seeds):
                        seed = cfg.seed_base + 1000 * k + 10 * int(N) + s
                        data = _estimate_sis(
                            operator=op,
                            sim_params=params,
                            timesteps=timesteps,
                            parallel=False,
                            num_samples=int(N),
                            noise_model=None,
                            seed=seed,
                            basis=cfg.basis,
                            basis_seed=cfg.basis_seed,
                        )
                        estimate = _sequence_data_to_dense(data)
                        comb_fit = DenseComb.fit(
                            estimate,
                            lam=float(cfg.fit_lam),
                            weight_tol=float(cfg.fit_weight_tol),
                        )
                        U_fit = comb_fit.to_matrix()
                        err_fit = float(rel_fro_error(U_fit, U_ex))
                        n_seq = len(data.sequences)

                        ms_res, n_obs = _constraint_ls_mean_sq_residual(
                            U_fit,
                            estimate,
                            weight_tol=float(cfg.fit_weight_tol),
                        )

                        state_errs: list[float] = []
                        state_trace_errs: list[float] = []
                        for ints in intervention_sets:
                            rho_ref = comb_ex.predict(ints)
                            rho_fit = comb_fit.predict(ints)
                            state_errs.append(
                                float(np.linalg.norm(rho_fit - rho_ref, ord="fro"))
                            )
                            state_trace_errs.append(_trace_distance(rho_fit, rho_ref))

                        if state_errs:
                            err_fit_pred_state_frob = float(np.mean(state_errs))
                            err_fit_pred_state_trace = float(
                                np.mean(state_trace_errs)
                            )
                            prefactor_fit_pred = float(
                                err_fit_pred_state_frob * np.sqrt(float(N))
                            )
                        else:
                            err_fit_pred_state_frob = float("nan")
                            err_fit_pred_state_trace = float("nan")
                            prefactor_fit_pred = float("nan")

                        trunc_frob: dict[int, float] = {}
                        trunc_trace: dict[int, float] = {}
                        for r in cfg.svd_trunc_ranks:
                            U_tr = _svd_truncate_frobenius(U_fit, r)
                            comb_tr = DenseComb(U_tr, list(estimate.timesteps))
                            te: list[float] = []
                            tt: list[float] = []
                            for ints in intervention_sets:
                                rho_ref = comb_ex.predict(ints)
                                rho_t = comb_tr.predict(ints)
                                te.append(
                                    float(np.linalg.norm(rho_t - rho_ref, ord="fro"))
                                )
                                tt.append(_trace_distance(rho_t, rho_ref))
                            trunc_frob[r] = float(np.mean(te)) if te else float("nan")
                            trunc_trace[r] = float(np.mean(tt)) if tt else float("nan")

                        trunc_frob_row = [trunc_frob[r] for r in cfg.svd_trunc_ranks]
                        trunc_trace_row = [trunc_trace[r] for r in cfg.svd_trunc_ranks]

                        print(
                            f"[sis] k={k} N={int(N)} seed={seed} "
                            f"n_obs={n_obs} mean_sq_res={ms_res:.3e} "
                            f"err_U={err_fit:.3e} "
                            f"err_state_frob={err_fit_pred_state_frob:.3e} "
                            f"trunc_frob[r={cfg.svd_trunc_ranks[-1]}]="
                            f"{trunc_frob.get(cfg.svd_trunc_ranks[-1], float('nan')):.3e}"
                        )

                        w.writerow(
                            [
                                variant["name"],
                                k,
                                int(N),
                                seed,
                                n_seq,
                                n_obs,
                                cfg.fit_lam,
                                cfg.fit_weight_tol,
                                ms_res,
                                err_fit,
                                err_fit_pred_state_frob,
                                err_fit_pred_state_trace,
                                prefactor_fit_pred,
                                *trunc_frob_row,
                                *trunc_trace_row,
                            ]
                        )

    summary_records: dict[tuple[str, int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with raw_path.open() as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            key = (row["variant"], int(row["k"]), int(row["N"]))
            for fk in float_keys:
                summary_records[key][fk].append(float(row[fk]))

    summary_header = (
        ["variant", "k", "N", "fit_lam", "fit_weight_tol"]
        + [f"mean_{fk}" for fk in float_keys]
        + [f"std_{fk}" for fk in float_keys]
    )

    with summary_path.open("w", newline="") as f_sum:
        w = csv.writer(f_sum)
        w.writerow(summary_header)
        for (variant, k, N), vals in sorted(summary_records.items()):
            row_out: list[object] = [variant, k, N, cfg.fit_lam, cfg.fit_weight_tol]
            for fk in float_keys:
                a = np.array(vals[fk], dtype=float)
                row_out.append(float(a.mean()))
            for fk in float_keys:
                a = np.array(vals[fk], dtype=float)
                row_out.append(float(a.std(ddof=0)))
            w.writerow(row_out)

    rows = list(csv.DictReader(summary_path.open()))
    print("Log-log slopes vs N for ||U_fit - U_ex||_F / ||U_ex||_F (excluding N = 16^k):")
    for variant in sorted({r["variant"] for r in rows}):
        for k in cfg.ks:
            rr = [r for r in rows if r["variant"] == variant and int(r["k"]) == k]
            rr = sorted(rr, key=lambda r: int(r["N"]))
            Ns_arr = np.array([int(r["N"]) for r in rr], dtype=int)
            n_exact = 16**k
            mask = Ns_arr != n_exact
            e_fit = np.array(
                [float(r["mean_err_fit_U_vs_exhaustive"]) for r in rr], dtype=float
            )
            print(
                f"  {variant:24s} k={k}  slope={_fit_loglog_slope(Ns_arr[mask], e_fit[mask]): .3f}"
            )
    print(f"Wrote: {raw_path}")
    print(f"Wrote: {summary_path}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark discrete _estimate (basis MC) + DenseComb.fit "
            "vs exhaustive comb Υ, with post-hoc SVD truncation diagnostics."
        ),
    )
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--ks", type=int, nargs="+", default=[2])
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--seed_base", type=int, default=12345)
    p.add_argument(
        "--out_dir",
        type=str,
        default="experiments_output/discrete_sis_fit_convergence",
    )
    p.add_argument("--prep_mixture_eps", type=float, default=0.1)
    p.add_argument("--basis", type=str, default="tetrahedral")
    p.add_argument("--basis_seed", type=int, default=12345)
    p.add_argument(
        "--proposal",
        type=str,
        default="local",
        choices=["uniform", "local", "mixture"],
    )
    p.add_argument("--solver", type=str, default="MCWF", choices=["TJM", "MCWF"])
    p.add_argument("--max_bond_dim", type=int, default=16)
    p.add_argument(
        "--fit_lam",
        type=float,
        default=1e-8,
        help="L2 regularization λ on vec(Υ) (passed to DenseComb.fit).",
    )
    p.add_argument(
        "--fit_weight_tol",
        type=float,
        default=1e-30,
        help="Treat sequence weights <= this as missing (passed to DenseComb.fit).",
    )
    p.add_argument(
        "--svd_trunc_ranks",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="SVD truncation ranks r for post-hoc prediction-error diagnostics.",
    )
    args = p.parse_args()
    return Config(
        L=args.L,
        ks=list(args.ks),
        dt=args.dt,
        n_seeds=args.n_seeds,
        seed_base=args.seed_base,
        out_dir=Path(args.out_dir),
        prep_mixture_eps=float(args.prep_mixture_eps),
        basis=str(args.basis),
        basis_seed=int(args.basis_seed) if args.basis_seed is not None else None,
        proposal=str(args.proposal),
        solver=str(args.solver),
        max_bond_dim=int(args.max_bond_dim),
        fit_lam=float(args.fit_lam),
        fit_weight_tol=float(args.fit_weight_tol),
        svd_trunc_ranks=list(args.svd_trunc_ranks),
    )


def main() -> None:
    cfg = parse_args()
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
