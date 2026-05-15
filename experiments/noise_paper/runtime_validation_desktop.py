"""Desktop wall-time sanity check for the alpha–kappa cost model (YAQS analog TDVP).

Compares physically equivalent unravelings (Pauli jumps vs. measurement/projector channels)
for dephasing, bit-flip, and depolarizing noise on a Heisenberg chain.

Projector jump names follow ``NoiseLibrary`` in
``mqt.yaqs.core.libraries.noise_library``:
Z basis: ``measure_0``, ``measure_1``; X: ``measure_x_0``, ``measure_x_1``;
Y: ``measure_y_0``, ``measure_y_1``.
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

# Cap parallel workers before importing YAQS (see simulator.available_cpus).
# if "YAQS_MAX_WORKERS" not in os.environ:
#    os.environ["YAQS_MAX_WORKERS"] = str(min(8, os.cpu_count() or 8))

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS, MPO
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z

# ---------------------------------------------------------------------------
# Defaults (benchmark setup)
# ---------------------------------------------------------------------------
L = 16
J = 1.0
H_FIELD = 1.0
GAMMA = 0.1
DT = 0.1
T_EVOL = 2.0
THRESHOLD = 1e-6
TRUNC_MODE = "discarded_weight"
ORDER = 2
SAMPLE_TIMESTEPS = False
TARGET_SE = 0.02
N_PILOT = 250
MAX_RUNTIME_TRAJ = 500
WARMUP_TRAJ = 10
TIMED_REPEATS = 3


def _verify_noise_names() -> None:
    """Fail fast if measurement operator names are renamed in YAQS."""
    for name in (
        "measure_0",
        "measure_1",
        "measure_x_0",
        "measure_x_1",
        "measure_y_0",
        "measure_y_1",
        "pauli_x",
        "pauli_y",
        "pauli_z",
    ):
        _ = NoiseModel.get_operator(name)


_verify_noise_names()


def build_observables(length: int) -> list[Observable]:
    """Local Z on the middle site plus max bond dimension."""
    return [Observable(Z(), [length // 2]), Observable("max_bond")]


def build_heisenberg(length: int, j: float, h: float) -> MPO:
    return MPO.heisenberg(length, j, j, j, h)


# --- Noise factories: A = Pauli jumps, B = measurement / projector unraveling ---


def noise_dephasing_jump(length: int, gamma: float) -> NoiseModel:
    return NoiseModel([{"name": "pauli_z", "sites": [i], "strength": gamma} for i in range(length)])


def noise_dephasing_meas(length: int, gamma: float) -> NoiseModel:
    return NoiseModel(
        [
            {"name": name, "sites": [i], "strength": 2.0 * gamma}
            for i in range(length)
            for name in ("measure_0", "measure_1")
        ]
    )


def noise_bitflip_jump(length: int, gamma: float) -> NoiseModel:
    return NoiseModel([{"name": "pauli_x", "sites": [i], "strength": gamma} for i in range(length)])


def noise_bitflip_meas(length: int, gamma: float) -> NoiseModel:
    return NoiseModel(
        [
            {"name": name, "sites": [i], "strength": 2.0 * gamma}
            for i in range(length)
            for name in ("measure_x_0", "measure_x_1")
        ]
    )


def noise_depol_jump(length: int, gamma: float) -> NoiseModel:
    return NoiseModel(
        [
            {"name": name, "sites": [i], "strength": gamma}
            for i in range(length)
            for name in ("pauli_x", "pauli_y", "pauli_z")
        ]
    )


def noise_depol_meas(length: int, gamma: float) -> NoiseModel:
    names = (
        "measure_x_0",
        "measure_x_1",
        "measure_y_0",
        "measure_y_1",
        "measure_0",
        "measure_1",
    )
    return NoiseModel(
        [{"name": name, "sites": [i], "strength": 2.0 * gamma} for i in range(length) for name in names]
    )


def extract_chi_max(sim_params: AnalogSimParams) -> float:
    max_bond_obs = sim_params.observables[1]
    if max_bond_obs.results is not None:
        return float(np.max(max_bond_obs.results))
    return 0.0


def extract_traj_samples(sim_params: AnalogSimParams) -> np.ndarray:
    obs = sim_params.observables[0]
    return np.real(obs.trajectories[:, -1]).astype(float)


def std_samples(samples: np.ndarray) -> float:
    if samples.size < 2:
        return 0.0
    return float(np.std(samples, ddof=1))


def n_req_from_std(std: float, target_se: float) -> int:
    if std <= 0.0 or not math.isfinite(std):
        return 1
    return max(1, int(math.ceil((std / target_se) ** 2)))


def make_sim_params(length: int, num_traj: int) -> AnalogSimParams:
    return AnalogSimParams(
        observables=build_observables(length),
        elapsed_time=T_EVOL,
        dt=DT,
        num_traj=num_traj,
        threshold=THRESHOLD,
        trunc_mode=TRUNC_MODE,
        order=ORDER,
        sample_timesteps=SAMPLE_TIMESTEPS,
        show_progress=False,
    )


def run_pilot(
    hamiltonian: MPO,
    length: int,
    noise_factory: Callable[[int, float], NoiseModel],
    num_traj: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    np.random.seed(seed)
    state = MPS(state="Neel", length=length)
    noise = noise_factory(length, GAMMA)
    sim_params = make_sim_params(length, num_traj)
    simulator.run(state, hamiltonian, sim_params, noise_model=noise, parallel=True)
    chi = extract_chi_max(sim_params)
    samples = extract_traj_samples(sim_params)
    return chi, std_samples(samples), samples


def warmup_scenario(
    hamiltonian: MPO,
    length: int,
    noise_a: Callable[[int, float], NoiseModel],
    noise_b: Callable[[int, float], NoiseModel],
    seed: int,
) -> None:
    np.random.seed(seed)
    for factory in (noise_a, noise_b):
        state = MPS(state="Neel", length=length)
        sim_params = make_sim_params(length, WARMUP_TRAJ)
        simulator.run(state, hamiltonian, sim_params, noise_model=factory(length, GAMMA), parallel=True)


def median_run_time_s(
    hamiltonian: MPO,
    length: int,
    noise_factory: Callable[[int, float], NoiseModel],
    num_traj: int,
    base_seed: int,
) -> float:
    times: list[float] = []
    for r in range(TIMED_REPEATS):
        np.random.seed(base_seed + r)
        state = MPS(state="Neel", length=length)
        sim_params = make_sim_params(length, num_traj)
        noise = noise_factory(length, GAMMA)
        t0 = time.perf_counter()
        simulator.run(state, hamiltonian, sim_params, noise_model=noise, parallel=True)
        times.append(time.perf_counter() - t0)
    return float(median(times))


@dataclass
class ScenarioResult:
    scenario: str
    workers: int
    chi_a: float
    chi_b: float
    alpha: float
    std_a: float
    std_b: float
    kappa_var: float
    n_req_a: int
    n_req_b: int
    n_run_a: int
    n_run_b: int
    p_a: int
    p_b: int
    predicted_ta_tb: float
    predicted_alpha_kappa: float
    time_a_median_s: float
    time_b_median_s: float
    measured_ta_tb: float
    predicted_preferred: str
    measured_preferred: str
    agreement: bool
    counts_note: str


def run_scenario(
    name: str,
    hamiltonian: MPO,
    length: int,
    workers: int,
    noise_a: Callable[[int, float], NoiseModel],
    noise_b: Callable[[int, float], NoiseModel],
    pilot_seed: int,
    time_seed: int,
) -> ScenarioResult:
    chi_a, std_a, _ = run_pilot(hamiltonian, length, noise_a, N_PILOT, pilot_seed)
    chi_b, std_b, _ = run_pilot(hamiltonian, length, noise_b, N_PILOT, pilot_seed + 1)

    alpha = chi_a / chi_b if chi_b > 0 else float("nan")
    if std_a > 0 and math.isfinite(std_a):
        kappa_var = (std_b / std_a) ** 2
    else:
        kappa_var = float("nan")

    n_req_a = n_req_from_std(std_a, TARGET_SE)
    n_req_b = n_req_from_std(std_b, TARGET_SE)

    n_run_a = n_req_a
    n_run_b = n_req_b
    counts_note = "using N_req as N_run"
    if max(n_run_a, n_run_b) > MAX_RUNTIME_TRAJ:
        scale = MAX_RUNTIME_TRAJ / max(n_run_a, n_run_b)
        n_run_a = max(20, round(n_run_a * scale))
        n_run_b = max(20, round(n_run_b * scale))
        counts_note = f"scaled to cap max(N_run) at {MAX_RUNTIME_TRAJ} (scale={scale:.4g})"

    p_a = min(workers, n_run_a)
    p_b = min(workers, n_run_b)

    predicted_ta_tb = ((n_run_a / p_a) * chi_a**3) / ((n_run_b / p_b) * chi_b**3) if chi_b > 0 else float("nan")
    predicted_alpha_kappa = ((alpha**3) / kappa_var) * (p_b / p_a) if kappa_var > 0 else float("nan")

    warmup_scenario(hamiltonian, length, noise_a, noise_b, time_seed)
    time_a = median_run_time_s(hamiltonian, length, noise_a, n_run_a, time_seed + 10)
    time_b = median_run_time_s(hamiltonian, length, noise_b, n_run_b, time_seed + 20)

    measured_ta_tb = time_a / time_b if time_b > 0 else float("nan")

    predicted_preferred = "A" if predicted_ta_tb < 1.0 else "B"
    measured_preferred = "A" if measured_ta_tb < 1.0 else "B"

    return ScenarioResult(
        scenario=name,
        workers=workers,
        chi_a=chi_a,
        chi_b=chi_b,
        alpha=alpha,
        std_a=std_a,
        std_b=std_b,
        kappa_var=kappa_var,
        n_req_a=n_req_a,
        n_req_b=n_req_b,
        n_run_a=n_run_a,
        n_run_b=n_run_b,
        p_a=p_a,
        p_b=p_b,
        predicted_ta_tb=predicted_ta_tb,
        predicted_alpha_kappa=predicted_alpha_kappa,
        time_a_median_s=time_a,
        time_b_median_s=time_b,
        measured_ta_tb=measured_ta_tb,
        predicted_preferred=predicted_preferred,
        measured_preferred=measured_preferred,
        agreement=predicted_preferred == measured_preferred,
        counts_note=counts_note,
    )


def result_to_row(res: ScenarioResult) -> dict[str, Any]:
    return {
        "scenario": res.scenario,
        "workers": res.workers,
        "L": L,
        "gamma": GAMMA,
        "dt": DT,
        "T": T_EVOL,
        "target_se": TARGET_SE,
        "N_pilot": N_PILOT,
        "chi_A": res.chi_a,
        "chi_B": res.chi_b,
        "alpha": res.alpha,
        "std_A": res.std_a,
        "std_B": res.std_b,
        "kappa_var": res.kappa_var,
        "N_req_A": res.n_req_a,
        "N_req_B": res.n_req_b,
        "N_run_A": res.n_run_a,
        "N_run_B": res.n_run_b,
        "P_A": res.p_a,
        "P_B": res.p_b,
        "predicted_TA_TB": res.predicted_ta_tb,
        "time_A_median_s": res.time_a_median_s,
        "time_B_median_s": res.time_b_median_s,
        "measured_TA_TB": res.measured_ta_tb,
        "predicted_preferred": res.predicted_preferred,
        "measured_preferred": res.measured_preferred,
        "agreement": res.agreement,
    }


CSV_COLUMNS = [
    "scenario",
    "workers",
    "L",
    "gamma",
    "dt",
    "T",
    "target_se",
    "N_pilot",
    "chi_A",
    "chi_B",
    "alpha",
    "std_A",
    "std_B",
    "kappa_var",
    "N_req_A",
    "N_req_B",
    "N_run_A",
    "N_run_B",
    "P_A",
    "P_B",
    "predicted_TA_TB",
    "time_A_median_s",
    "time_B_median_s",
    "measured_TA_TB",
    "predicted_preferred",
    "measured_preferred",
    "agreement",
]


def print_table(results: list[ScenarioResult]) -> None:
    print("\n=== Runtime validation (A = Pauli jumps, B = measurements) ===\n")
    for res in results:
        print(f"[{res.scenario}] {res.counts_note}")
        print(
            f"  chi_A={res.chi_a:.3f} chi_B={res.chi_b:.3f} alpha={res.alpha:.4f}  "
            f"std_A={res.std_a:.5f} std_B={res.std_b:.5f} kappa_var={res.kappa_var:.4f}"
        )
        print(
            f"  N_req_A={res.n_req_a} N_req_B={res.n_req_b}  N_run_A={res.n_run_a} N_run_B={res.n_run_b}  "
            f"P_A={res.p_a} P_B={res.p_b}"
        )
        print(
            f"  predicted T_A/T_B={res.predicted_ta_tb:.4f}  "
            f"(alpha^3/kappa_var)*(P_B/P_A)={res.predicted_alpha_kappa:.4f}"
        )
        print(
            f"  median time A={res.time_a_median_s:.3f}s B={res.time_b_median_s:.3f}s  "
            f"measured T_A/T_B={res.measured_ta_tb:.4f}"
        )
        print(
            f"  preferred (predicted)={res.predicted_preferred}  "
            f"(measured)={res.measured_preferred}  agreement={res.agreement}\n"
        )

    # Compact summary
    hdr = (
        f"{'scenario':<14} {'pred T_A/TB':>12} {'meas T_A/TB':>12} "
        f"{'pred':>5} {'meas':>5} {'ok':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for res in results:
        print(
            f"{res.scenario:<14} {res.predicted_ta_tb:12.4f} {res.measured_ta_tb:12.4f} "
            f"{res.predicted_preferred:>5} {res.measured_preferred:>5} {str(res.agreement):>5}"
        )


def main() -> None:
    workers = int(os.environ.get("YAQS_MAX_WORKERS", str(min(8, os.cpu_count() or 8))))
    hamiltonian = build_heisenberg(L, J, H_FIELD)

    scenarios: list[tuple[str, Callable[[int, float], NoiseModel], Callable[[int, float], NoiseModel]]] = [
        ("dephasing", noise_dephasing_jump, noise_dephasing_meas),
        ("bitflip", noise_bitflip_jump, noise_bitflip_meas),
        ("depolarizing", noise_depol_jump, noise_depol_meas),
    ]

    results: list[ScenarioResult] = []
    rows: list[dict[str, Any]] = []
    for idx, (name, na, nb) in enumerate(scenarios):
        res = run_scenario(name, hamiltonian, L, workers, na, nb, pilot_seed=1000 + 50 * idx, time_seed=5000 + 50 * idx)
        results.append(res)
        rows.append(result_to_row(res))

    out_path = Path(__file__).resolve().parent / "runtime_validation_desktop.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print_table(results)
    print(f"Wrote {out_path}")
    print(
        "\nThis is a desktop wall-time sanity check. It validates the qualitative ordering "
        "of the cost model, not exact implementation-independent prefactors."
    )


if __name__ == "__main__":
    main()
