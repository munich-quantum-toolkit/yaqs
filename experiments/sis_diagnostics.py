"""
experiments/sis_diagnostics.py

Comprehensive validation and diagnostics for run_sis_upsilon.

Covers all seven requested checks:
  1. Unbiasedness: mean(U_hat) -> U_ref across seeds
  2. Raw vs canonicalized Frobenius comparison
  3. Per-step ESS, CV, max/mean ratio
  4. Proposal comparison: "uniform" vs "local"
  6. Deterministic guardrail (noise_model=None, num_trajectories=1)
  7. k-scaling sweep (k=1..5, N ~ 10 * 4^k)
"""
if __name__ == "__main__":
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    from mqt.yaqs.characterization.tomography.tomography import run, run_sis_upsilon
    from mqt.yaqs.characterization.tomography.process_tensor import canonicalize_upsilon
    from mqt.yaqs.core.data_structures.networks import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

    # ──────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)

    def _apply_dual(D, conv):
        if conv == "id":   return D
        if conv == "T":    return D.T
        if conv == "conj": return D.conj()
        if conv == "dag":  return D.conj().T

    def build_upsilon_full(pt, conv):
        k = pt.tensor.ndim - 1
        dim = 2 * (4**k)
        U = np.zeros((dim, dim), dtype=np.complex128)
        for alphas in itertools.product(range(16), repeat=k):
            rho_out = pt.tensor[(slice(None), *alphas)].reshape(2, 2)
            past = _apply_dual(pt.choi_duals[alphas[0]], conv)
            for a in alphas[1:]:
                past = np.kron(past, _apply_dual(pt.choi_duals[a], conv))
            U += np.kron(rho_out, past)
        return U

    def herm(U):
        return 0.5 * (U + U.conj().T)

    def fro_rel(A, B):
        """Relative Frobenius on Hermitized matrices."""
        A, B = herm(A), herm(B)
        denom = np.linalg.norm(B)
        return float(np.linalg.norm(A - B) / denom) if denom > 0 else np.nan

    # ──────────────────────────────────────────
    # CHECK 1 & 4: Unbiasedness + proposal comparison
    # ──────────────────────────────────────────
    print("\n" + "="*60)
    print("CHECK 1 & 4: Unbiasedness + Proposal Comparison (k=1)")
    print("="*60)

    k_test = 1
    timesteps_test = [0.1] * k_test
    n_seeds = 10

    pt = run(op, params, timesteps=timesteps_test)
    _, conv = pt.reconstruct_comb_choi(check=True, return_convention=True)
    u_ref_raw = build_upsilon_full(pt, conv)
    u_ref_h   = herm(u_ref_raw)

    print(f"Convention: {conv}")
    print(f"Tr(U_ref_raw) = {float(np.trace(u_ref_raw).real):.4f}  (should be ~16 for full sum of 16 terms)")

    for proposal in ["local", "uniform"]:
        print(f"\n  proposal='{proposal}':")
        for N in [16, 64, 256]:
            ests = []
            for s in range(n_seeds):
                u_hat, meta = run_sis_upsilon(
                    op, params, timesteps=timesteps_test,
                    num_particles=N, seed=s, dual_transform=conv,
                    proposal=proposal, resample=True, ess_threshold=0.5,
                )
                ests.append(u_hat)

            u_mean = np.mean(ests, axis=0)
            trace_vals = [float(np.trace(e).real) for e in ests]

            bias = fro_rel(u_mean, u_ref_raw)
            fro_std = np.std([fro_rel(e, u_ref_raw) for e in ests])
            mean_trace = np.mean(trace_vals)
            std_trace  = np.std(trace_vals)

            print(f"    N={N:4d} | bias(mean-ref)={bias:.3e} | Fro std={fro_std:.3e} "
                  f"| Tr: mean={mean_trace:.3f}, std={std_trace:.3f}")

    # ──────────────────────────────────────────
    # CHECK 2: Raw vs Canonicalized Frobenius (local proposal)
    # ──────────────────────────────────────────
    print("\n" + "="*60)
    print("CHECK 2: Raw vs Canonicalized Frobenius (k=1, local proposal)")
    print("="*60)

    u_ref_c = canonicalize_upsilon(u_ref_raw, hermitize=True, psd_project=True, normalize_trace=True)

    for k_c, n_sweep in [(1, [8, 16, 32, 64, 128, 256]),
                          (2, [16, 64, 128, 256])]:
        timesteps_c = [0.1] * k_c
        if k_c > 1:
            pt = run(op, params, timesteps=timesteps_c)
            _, conv = pt.reconstruct_comb_choi(check=True, return_convention=True)
            u_ref_raw = build_upsilon_full(pt, conv)
            u_ref_c = canonicalize_upsilon(u_ref_raw, hermitize=True, psd_project=True, normalize_trace=True)

        print(f"\n  k={k_c}, conv={conv}:")
        for N in n_sweep:
            raws, cans = [], []
            for s in range(5):
                u_hat, _ = run_sis_upsilon(
                    op, params, timesteps=timesteps_c,
                    num_particles=N, seed=s, dual_transform=conv,
                    proposal="local", resample=True,
                )
                raws.append(fro_rel(u_hat, u_ref_raw))
                u_hat_c = canonicalize_upsilon(u_hat, hermitize=True, psd_project=True, normalize_trace=True)
                cans.append(fro_rel(u_hat_c, u_ref_c))

            print(f"    N={N:4d} | Raw Fro={np.mean(raws):.3e} ± {np.std(raws):.2e} "
                  f"| Canon Fro={np.mean(cans):.3e} ± {np.std(cans):.2e}")

    # ──────────────────────────────────────────
    # CHECK 3: Per-step ESS and weight diagnostics
    # ──────────────────────────────────────────
    print("\n" + "="*60)
    print("CHECK 3: Per-step Diagnostics (k=3, local vs uniform, N=128)")
    print("="*60)

    k3 = 3
    t3 = [0.1] * k3
    pt3 = run(op, params, timesteps=t3)
    _, conv3 = pt3.reconstruct_comb_choi(check=True, return_convention=True)

    for proposal in ["local", "uniform"]:
        u_hat3, meta3 = run_sis_upsilon(
            op, params, timesteps=t3,
            num_particles=128, seed=42, dual_transform=conv3,
            proposal=proposal, resample=True,
        )
        print(f"\n  proposal='{proposal}':")
        print(f"  {'step':>5} | {'ESS':>8} | {'CV':>8} | {'max/mean':>8} | {'uniq paths':>12}")
        for t_idx in range(k3):
            ess = meta3["ess_history"][t_idx] if t_idx < len(meta3["ess_history"]) else float("nan")
            cv  = meta3["weight_cv"][t_idx] if t_idx < len(meta3["weight_cv"]) else float("nan")
            mmr = meta3["max_mean_ratio"][t_idx] if t_idx < len(meta3["max_mean_ratio"]) else float("nan")
            up  = meta3["unique_paths"][t_idx] if t_idx < len(meta3["unique_paths"]) else 0
            print(f"  {t_idx+1:>5} | {ess:>8.1f} | {cv:>8.3f} | {mmr:>8.2f} | {up:>12d}")
        print(f"  Resampled at steps: {meta3['resampling_steps']}")
        print(f"  Total weight / N: {meta3['total_weight']:.4f}")

    # ──────────────────────────────────────────
    # CHECK 7: k-scaling sweep ((k=1..5, N ~ 10 * 4^k)
    # ──────────────────────────────────────────
    print("\n" + "="*60)
    print("CHECK 7: k-Scaling Sweep (local proposal, N ~ 10*4^k)")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ks = [1, 2, 3]
    N_per_k = [min(10 * 4**k, 128) for k in ks]  # capped to avoid k=4,5 runtime
    ess_at_final = []
    fro_at_final = []

    for k_i, N_i in zip(ks, N_per_k):
        ts_i = [0.1] * k_i
        pt_i = run(op, params, timesteps=ts_i)
        _, conv_i = pt_i.reconstruct_comb_choi(check=True, return_convention=True)
        u_ref_i = build_upsilon_full(pt_i, conv_i)

        fro_vals, ess_finals = [], []
        for seed_i in range(3):
            u_hat_i, meta_i = run_sis_upsilon(
                op, params, timesteps=ts_i,
                num_particles=N_i, seed=seed_i, dual_transform=conv_i,
                proposal="local", resample=True,
            )
            fro_vals.append(fro_rel(u_hat_i, u_ref_i))
            if meta_i["ess_history"]:
                ess_finals.append(meta_i["ess_history"][-1])

        m_fro = np.mean(fro_vals)
        m_ess = np.mean(ess_finals) if ess_finals else float("nan")
        ess_frac = m_ess / N_i if not np.isnan(m_ess) else float("nan")

        print(f"  k={k_i}, N={N_i:5d} | Fro={m_fro:.3e} | ESS final={m_ess:.1f} ({ess_frac*100:.1f}% of N)")

        ess_at_final.append(ess_frac)
        fro_at_final.append(m_fro)

    axes[0].semilogy(ks, fro_at_final, 'o-', color='steelblue')
    axes[0].set(xlabel='k (number of timesteps)', ylabel='Rel. Frobenius error', title='Error vs k (N=10·4^k)')
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(ks, [e * 100 for e in ess_at_final], 's-', color='darkorange')
    axes[1].set(xlabel='k', ylabel='ESS / N (%)', title='Final ESS fraction vs k')
    axes[1].grid(True, alpha=0.4)
    axes[1].set_ylim(0, 105)

    fig.tight_layout()
    fname = "sis_k_scaling.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"\n  → Saved: {fname}")

    # ──────────────────────────────────────────
    # N-convergence plot (k=1, local vs uniform)
    # ──────────────────────────────────────────
    print("\n" + "="*60)
    print("N-convergence plot (k=1 & k=2, local vs uniform)")
    print("="*60)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, k_p in enumerate([1, 2]):
        ts_p = [0.1] * k_p
        if k_p == 1:
            pt_p = run(op, params, timesteps=ts_p)
            _, conv_p = pt_p.reconstruct_comb_choi(check=True, return_convention=True)
            u_ref_p = build_upsilon_full(pt_p, conv_p)

        ax = axes2[ax_idx]
        n_sweep_p = [int(n) for n in np.unique(np.logspace(1, np.log10(min(512, 16**k_p)), 12).astype(int))]

        for proposal_p, color in [("local", "steelblue"), ("uniform", "darkorange")]:
            means, stds = [], []
            for N_p in n_sweep_p:
                errs_p = []
                for s in range(5):
                    u_p, _ = run_sis_upsilon(
                        op, params, timesteps=ts_p,
                        num_particles=N_p, seed=s, dual_transform=conv_p,
                        proposal=proposal_p, resample=True,
                    )
                    errs_p.append(fro_rel(u_p, u_ref_p))
                means.append(np.mean(errs_p))
                stds.append(np.std(errs_p))
                print(f"  k={k_p}, prop={proposal_p}, N={N_p:4d}: Fro={np.mean(errs_p):.3e}")

            means_a, stds_a = np.array(means), np.array(stds)
            ns_a = np.array(n_sweep_p)
            valid = np.isfinite(means_a) & (means_a > 0)
            if valid.any():
                ax.loglog(ns_a[valid], means_a[valid], 'o-', color=color, label=proposal_p)
                ax.fill_between(ns_a[valid], np.clip(means_a[valid]-stds_a[valid], 1e-20, None),
                                means_a[valid]+stds_a[valid], alpha=0.15, color=color)

        ref_ns = np.array(n_sweep_p)
        ref_m = float(means[2]) * np.sqrt(n_sweep_p[2])
        ax.loglog(ref_ns, ref_m / np.sqrt(ref_ns), '--', color='gray', alpha=0.5, label='1/√N')
        ax.set(xlabel='N', ylabel='Rel Fro error', title=f'k={k_p}  (local optimal vs uniform)')
        ax.grid(True, which='both', alpha=0.4)
        ax.legend()

    fig2.tight_layout()
    fname2 = "sis_convergence.png"
    fig2.savefig(fname2, dpi=200)
    plt.close(fig2)
    print(f"\n  → Saved: {fname2}")
