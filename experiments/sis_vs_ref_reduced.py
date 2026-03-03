"""experiments/sis_vs_ref_reduced.py
Compare SIS vs full-enumeration reference using reduced_upsilon (m=1, i.e. 8x8)
for k = 1 .. 4. This makes the metric constant-size independent of k.

Usage: python experiments/sis_vs_ref_reduced.py
Outputs: sis_k_reduced.txt  +  sis_k_reduced.png
"""
if __name__ == "__main__":
    import itertools, copy
    import numpy as np
    import matplotlib.pyplot as plt

    from mqt.yaqs.characterization.tomography.tomography import run, run_sis_upsilon
    from mqt.yaqs.characterization.tomography.process_tensor import (
        reduced_upsilon, canonicalize_upsilon, comb_qmi_from_upsilon_dense,
    )
    from mqt.yaqs.core.data_structures.networks import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

    # ── System ─────────────────────────────────────────────────────────────────
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    M = 1          # keep_last_m = 1  →  always 8×8 metric
    N_SEEDS = 5
    DT = 0.1

    # ── Build full-enumeration reference for k=1..4 ───────────────────────────
    def _dual(D, conv):
        return {"id": D, "T": D.T, "conj": D.conj(), "dag": D.conj().T}[conv]

    def build_ref(pt, conv):
        k = pt.tensor.ndim - 1
        U = np.zeros((2 * 4**k,) * 2, dtype=np.complex128)
        for alphas in itertools.product(range(16), repeat=k):
            rho = pt.tensor[(slice(None), *alphas)].reshape(2, 2)
            P = _dual(pt.choi_duals[alphas[0]], conv)
            for a in alphas[1:]:
                P = np.kron(P, _dual(pt.choi_duals[a], conv))
            U += np.kron(rho, P)
        return U

    def hfro(A, B):
        """Relative Frobenius error with nan protection."""
        A = np.nan_to_num(0.5*(A+A.conj().T), nan=0.0)
        B = 0.5*(B+B.conj().T)
        d = np.linalg.norm(B)
        return float(np.linalg.norm(A-B)/d) if d > 0 else np.nan

    results = {}  # k -> {N: {"fro_raw":..., "fro_red":..., "qmi_err":..., "ess":...}}

    print(f"{'k':>2} {'N':>6} {'Fro_raw':>12} {'Fro_red(m=1)':>14} {'|ΔQMI|':>10} "
          f"{'alpha_ent':>10} {'unique_paths':>14}")

    KS = [1, 2, 3, 4]
    NS_by_k = {1: [16, 32, 64, 128, 256], 2: [32, 64, 128, 256], 3: [64, 128, 256], 4: [128, 256]}

    for k in KS:
        ts = [DT] * k
        pt = run(op, params, timesteps=ts)
        _, conv = pt.reconstruct_comb_choi(check=True, return_convention=True)
        U_ref = build_ref(pt, conv)
        # Canonicalize reference for QMI
        U_ref_c = canonicalize_upsilon(U_ref, hermitize=True, psd_project=True, normalize_trace=True)
        qmi_ref = comb_qmi_from_upsilon_dense(U_ref_c, check_psd=False)
        # Reduce reference
        U_ref_red = reduced_upsilon(U_ref, k=k, keep_last_m=M)

        results[k] = {}
        for N in NS_by_k[k]:
            fros_raw, fros_red, qmi_errs, ents, ups = [], [], [], [], []
            for s in range(N_SEEDS):
                u, m = run_sis_upsilon(op, params, timesteps=ts,
                    num_particles=N, seed=s, dual_transform=conv,
                    proposal="local", stratify_step1=True, parallel=True)

                # Raw Frobenius (unreduced)
                fros_raw.append(hfro(u, U_ref))

                # Reduced Frobenius (always 8x8)
                u_red = reduced_upsilon(u, k=k, keep_last_m=M)
                fros_red.append(hfro(u_red, U_ref_red))

                # QMI error (on canonicalized)
                u_c = canonicalize_upsilon(u, hermitize=True, psd_project=True, normalize_trace=True)
                try:
                    qmi_est = comb_qmi_from_upsilon_dense(u_c, check_psd=False)
                    qmi_errs.append(abs(qmi_est - qmi_ref))
                except Exception:
                    qmi_errs.append(float("nan"))

                ents.append(np.mean(m["alpha_entropy"]) if m["alpha_entropy"] else float("nan"))
                ups.append(m["unique_paths"][-1] if m["unique_paths"] else 0)

            results[k][N] = dict(
                fro_raw=np.nanmean(fros_raw), fro_red=np.nanmean(fros_red),
                qmi_err=np.nanmean(qmi_errs), ent=np.nanmean(ents), up=np.mean(ups),
            )
            r = results[k][N]
            print(f"{k:>2} {N:>6} {r['fro_raw']:>12.4e} {r['fro_red']:>14.4e} "
                  f"{r['qmi_err']:>10.4e} {r['ent']:>10.3f} {r['up']:>14.1f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}

    for k in KS:
        ns = sorted(results[k].keys())
        fros_raw = [results[k][N]["fro_raw"] for N in ns]
        fros_red = [results[k][N]["fro_red"] for N in ns]
        qmi_errs = [results[k][N]["qmi_err"] for N in ns]

        axes[0].loglog(ns, fros_raw, color=colors[k], marker=markers[k], label=f"k={k}")
        axes[1].loglog(ns, fros_red, color=colors[k], marker=markers[k], label=f"k={k}")
        axes[2].loglog(ns, qmi_errs, color=colors[k], marker=markers[k], label=f"k={k}")

    # 1/sqrt(N) reference line
    ns_ref = np.array([16, 64, 256])
    axes[0].loglog(ns_ref, 2e0 / np.sqrt(ns_ref), 'k--', alpha=0.4, label='1/√N')
    axes[1].loglog(ns_ref, 5e-1 / np.sqrt(ns_ref), 'k--', alpha=0.4, label='1/√N')
    axes[2].loglog(ns_ref, 2e-1 / np.sqrt(ns_ref), 'k--', alpha=0.4, label='1/√N')

    for ax, title in zip(axes, [
        f"Global Fro (2·4^k × 2·4^k)",
        f"Reduced Fro (8×8, m={M})",
        "|ΔQMI| (bits)",
    ]):
        ax.set(xlabel="N particles", title=title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.4)

    axes[0].set_ylabel("Relative Frobenius error")
    fig.suptitle("SIS (local+stratify) vs Reference  —  k=1..4, m=1 reduction")
    fig.tight_layout()
    fig.savefig("sis_k_reduced.png", dpi=180)
    plt.close(fig)
    print("\nSaved sis_k_reduced.png")

    # ── Text report ────────────────────────────────────────────────────────────
    lines = ["k-Scaling Report: SIS vs reference (reduced_upsilon m=1)"]
    lines.append(f"{'k':>2} {'N':>6} {'Fro_raw':>12} {'Fro_red':>12} {'|ΔQMI|':>10}")
    for k in KS:
        for N in sorted(results[k].keys()):
            r = results[k][N]
            lines.append(f"{k:>2} {N:>6} {r['fro_raw']:>12.4e} {r['fro_red']:>12.4e} {r['qmi_err']:>10.4e}")
    with open("sis_k_reduced.txt", "w") as f:
        f.write("\n".join(lines))
    print("Saved sis_k_reduced.txt")
