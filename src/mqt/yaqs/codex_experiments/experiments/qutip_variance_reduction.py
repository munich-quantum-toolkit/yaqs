# Write a ready-to-run QuTiP script that compares three unravelings
# (1) Standard "unitary jump" with L = sqrt(gamma) X  (digital MCWF)
# (2) Projector jumps L_pm = sqrt(gamma/2) (I ± X)     (counting MCWF with projectors)
# (3) Analog rotations U = exp(i theta X) at each small step, with theta drawn
#     either from a Gaussian N(0, sigma^2) or a symmetric two-point law ±arcsin(sqrt(q)),
#     tuned to match the same Pauli-X channel in expectation.
#
# The script computes <Z> and the per-trajectory variance Var[<Z>] at t_final,
# and prints MC estimates alongside the closed-form predictions.
#
# Save as /mnt/data/qutip_unravelings_variance.py


import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal, Tuple, Optional
try:
    from qutip import basis, qeye, sigmax, sigmaz, mcsolve, sesolve, Options, expect
except Exception as e:
    raise SystemExit("This script requires QuTiP. Please install it with `pip install qutip`.")


@dataclass
class Stats:
    mu_hat: float
    var_hat: float
    mu_th: float
    var_th: float
    unravel: str
    gamma: float
    t: float
    ntraj: int
    note: str = ""
    mu_t_mc: Optional[np.ndarray] = None
    mu_t_th: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None


def _extract_runs_states(res):
    # QuTiP 5.x uses result.runs_states; QuTiP 4.x stores in result.states (list of lists)
    runs = getattr(res, "runs_states", None)
    if runs is None:
        runs = res.states
    return runs


def theory_X_unravel(gamma: float, t: float) -> Tuple[float, float]:
    mu = np.exp(-2 * gamma * t)
    var = 1.0 - np.exp(-4 * gamma * t)
    return mu, var


def theory_projector_unravel(gamma: float, t: float) -> Tuple[float, float]:
    mu = np.exp(-2 * gamma * t)
    var = np.exp(-2 * gamma * t) * (1.0 - np.exp(-2 * gamma * t))
    return mu, var


def theory_analog_gauss(gamma: float, t: float) -> Tuple[float, float]:
    # Gaussian angles theta ~ N(0, sigma^2) with sigma^2 = gamma * dt at each of n steps = t / dt
    # => E[cos 2theta] = exp(-2 sigma^2) per step; mean over n steps: exp(-2 gamma t)
    # E[cos 4theta] = exp(-8 sigma^2) per step -> variance: 0.5 + 0.5 exp(-8 gamma t) - exp(-4 gamma t)
    mu = np.exp(-2 * gamma * t)
    var = 0.5 + 0.5 * np.exp(-8 * gamma * t) - np.exp(-4 * gamma * t)
    return mu, var


def theory_analog_twopoint(gamma: float, t: float, dt: float) -> Tuple[float, float]:
    # Two-point ±theta0 with sin^2(theta0) = q, where (1 - 2q) = exp(-2 gamma dt)
    # Per step: E[cos 2theta] = 1 - 2q = exp(-2 gamma dt); E[cos 4theta] = cos 4theta0 = 2(1-2q)^2 - 1
    # After n steps: mu = (1 - 2q)^n = exp(-2 gamma t)
    # var = 0.5 + 0.5 [cos 4theta0]^n - [ (1 - 2q)^2 ]^n
    q = 0.5 * (1.0 - np.exp(-2 * gamma * dt))
    cos4 = 2.0 * (1.0 - 2.0 * q)**2 - 1.0
    n = int(round(t / dt))
    mu = np.exp(-2 * gamma * t)
    var = 0.5 + 0.5 * (cos4 ** n) - np.exp(-4 * gamma * t)
    return mu, var


def run_mcwf(unravel: Literal["X", "proj", "analog-2pt", "analog-gauss"], gamma=1.0, t_final=2.0, dt=0.01, ntraj=2000, seed=None, theta0: float = 0.2, sigma: float = 0.2) -> Stats:
    I, X, Z = qeye(2), sigmax(), sigmaz()
    H = 0 * Z  # no Hamiltonian
    if unravel == "X":
        c_ops = [np.sqrt(gamma) * X]
        mu_th, var_th = theory_X_unravel(gamma, t_final)
        note = "MCWF unitary-jump with L=sqrt(gamma) X"
    elif unravel == "proj":
        c_ops = [np.sqrt(gamma / 2.0) * (I + X), np.sqrt(gamma / 2.0) * (I - X)]
        mu_th, var_th = theory_projector_unravel(gamma, t_final)
        note = "MCWF projector-jumps with L_±=sqrt(gamma/2)(I±X)"
    elif unravel == "analog-2pt":
        c_ops, _ = build_unitary_cops_2pt(X, gamma, theta0)
        note = "Analog via mcsolve: unitary jumps at ±θ0 with λ chosen s.t. λ sin^2θ0=γ"
        mu_th, var_th = theory_analog_twopoint(gamma, t_final, dt)
    elif unravel == "analog-gauss":
        c_ops, _ = build_unitary_cops_gauss(X, gamma, sigma, M=31, theta_max=4 * sigma)
        note = "Analog via mcsolve: unitary jumps with θ~N(0,σ^2) (discretized); λ chosen s.t. λ E[sin^2θ]=γ"
        mu_th, var_th = theory_analog_gauss(gamma, t_final)
    else:
        raise ValueError("unravel must be 'X', 'proj', 'analog-2pt', or 'analog-gauss'")

    psi0 = basis(2, 0)  # |0>
    times = np.linspace(0.0, t_final, int(round(t_final / dt)) + 1)
    opts = Options(keep_runs_results=True, store_states=True, progress_bar=None)
    res = mcsolve(H, psi0, times, c_ops=c_ops, e_ops=None, ntraj=ntraj, options=opts, progress_bar=None, seeds=None)
    runs = _extract_runs_states(res)
    # Time-series Z expectation for each trajectory
    z_trajs = np.array([[float(expect(Z, traj[t_idx]).real) for t_idx in range(len(times))] for traj in runs])
    z_final = z_trajs[:, -1]
    mu_hat = float(z_final.mean())
    var_hat = float(z_final.var(ddof=1))
    mu_t_mc = z_trajs.mean(axis=0)
    mu_t_th = np.exp(-2 * gamma * times)
    return Stats(mu_hat, var_hat, mu_th, var_th, unravel, gamma, t_final, ntraj, note=note,
                 mu_t_mc=mu_t_mc, mu_t_th=mu_t_th, times=times)


def build_unitary_cops_2pt(X, gamma: float, theta0: float):
    s = np.sin(theta0) ** 2
    if s <= 0.0:
        raise ValueError("theta0 too small; sin^2(theta0)=0.")
    lam = gamma / s
    Uplus = (1j * theta0 * X).expm()
    Uminus = (-1j * theta0 * X).expm()
    c_ops = [np.sqrt(lam / 2.0) * Uplus, np.sqrt(lam / 2.0) * Uminus]
    return c_ops, {"lambda_total": lam, "s": s, "theta0": theta0}


def build_unitary_cops_gauss(X, gamma: float, sigma: float, M: int = 31, theta_max: float | None = None):
    if theta_max is None:
        theta_max = 4.0 * sigma
    thetas_pos = np.linspace(0.0, theta_max, (M + 1) // 2)
    thetas = np.concatenate([-thetas_pos[:0:-1], thetas_pos])
    w = np.exp(-0.5 * (thetas / sigma) ** 2)
    w /= w.sum()
    w = 0.5 * (w + w[::-1])
    s = float(np.sum(w * np.sin(thetas) ** 2))
    if s <= 1e-12:
        raise ValueError("s=E[sin^2 θ] too small; increase sigma or theta_max/M.")
    lam = gamma / s
    c_ops = []
    for wk, th in zip(w, thetas):
        if wk <= 0.0:
            continue
        U = (1j * th * X).expm()
        c_ops.append(np.sqrt(lam * wk) * U)
    return c_ops, {"lambda_total": lam, "s": s, "theta_grid": thetas, "weights": w}

def plot_all(stats_list):
    # Assume all share the same time grid and exact curve
    times = stats_list[0].times
    mu_t_th = stats_list[0].mu_t_th
    plt.figure(figsize=(8, 5))
    plt.plot(times, mu_t_th, label="Exact e^{-2γt}", color="black", linewidth=2.0)
    colors = {
        "X": "tab:blue",
        "proj": "tab:orange",
        "analog-gauss": "tab:green",
        "analog-2pt": "tab:red",
    }
    labels = {
        "X": "MC X-unravel",
        "proj": "MC proj-jumps",
        "analog-gauss": "MC analog (gauss)",
        "analog-2pt": "MC analog (±2pt)",
    }
    for s in stats_list:
        plt.plot(s.times, s.mu_t_mc, label=labels.get(s.unravel, s.unravel), color=colors.get(s.unravel))
    plt.xlabel("time")
    plt.ylabel("⟨Z⟩")
    plt.title("⟨Z⟩(t): exact vs MC (X, proj, analog)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("qutip_unravelings_z_vs_time.png", dpi=150)
    try:
        plt.show()
    except Exception:
        pass


def demo():
    gamma = 0.0
    t_final = 20.0
    dt = 0.01
    ntraj = 2000
    print("=== Comparing unravelings for X-channel (no Hamiltonian), observable Z ===")
    stats = []
    for unr in ["X", "proj", "analog-gauss", "analog-2pt"]:
        s = run_mcwf(unravel=unr, gamma=gamma, t_final=t_final, dt=dt, ntraj=ntraj, seed=None)
        stats.append(s)
    for s in stats:
        print(f"{s.unravel:>5s}:  ⟨Z⟩ MC={s.mu_hat:.4f} (th {s.mu_th:.4f})   Var MC={s.var_hat:.4f} (th {s.var_th:.4f})  ntraj={s.ntraj}")
    plot_all(stats)


if __name__ == "__main__":
    demo()


