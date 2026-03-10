import pytest
import numpy as np

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

from mqt.yaqs.characterization.tomography.tomography import (
    run_mc_upsilon, 
    run_mc_upsilon_mpo,
    run_sis_upsilon,
    run_sis_upsilon_mpo
)
from mqt.yaqs.characterization.tomography.process_tensor_mpo import (
    rank1_upsilon_mpo_term,
    upsilon_mpo_to_dense,
)

# --- Helpers ---

def random_rho():
    np.random.seed(42)
    rho = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    rho = rho @ rho.conj().T
    return rho / np.trace(rho)

def random_complex_matrix(dim):
    return np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)

def small_hamiltonian(length=1):
    return MPO.hamiltonian(length=length, one_body=[(1.0, "X")], two_body=[])

def rel_err(A, B):
    return np.linalg.norm(A - B) / max(np.linalg.norm(A), 1e-15)

def get_sim_params():
    return AnalogSimParams(elapsed_time=0.1, dt=0.01, solver="MCWF", show_progress=False)

# --- MPO Algebra Tests ---

@pytest.mark.parametrize("k", [1, 2, 3])
def test_rank1_mpo_vs_kron(k):
    np.random.seed(42 + k)
    rho = random_rho()
    ops = [random_complex_matrix(4) for _ in range(k)]
    w = 1.23

    mpo = rank1_upsilon_mpo_term(rho, ops, weight=w)
    dense_mpo = upsilon_mpo_to_dense(mpo)
    
    dense_ref = rho
    for op in ops:
        dense_ref = np.kron(dense_ref, op)
    dense_ref *= w

    assert rel_err(dense_mpo, dense_ref) < 1e-12

def test_mpo_addition_matches_dense():
    np.random.seed(111)
    rho1, rho2 = random_rho(), random_rho()
    ops1 = [random_complex_matrix(4), random_complex_matrix(4)]
    ops2 = [random_complex_matrix(4), random_complex_matrix(4)]
    
    mpo1 = rank1_upsilon_mpo_term(rho1, ops1, weight=1.0)
    mpo2 = rank1_upsilon_mpo_term(rho2, ops2, weight=0.5)
    
    mpo_added = mpo1 + mpo2
    dense_added = upsilon_mpo_to_dense(mpo_added)
    
    d1 = upsilon_mpo_to_dense(mpo1)
    d2 = upsilon_mpo_to_dense(mpo2)
    
    assert rel_err(dense_added, d1 + d2) < 1e-12

def test_mpo_sum_matches_dense():
    np.random.seed(222)
    mpos = []
    denses = []
    for _ in range(5):
        rho = random_rho()
        ops = [random_complex_matrix(4), random_complex_matrix(4)]
        w = np.random.rand()
        m = rank1_upsilon_mpo_term(rho, ops, weight=w)
        mpos.append(m)
        denses.append(upsilon_mpo_to_dense(m))
        
    mpo_summed = MPO.mpo_sum(mpos)
    dense_summed = sum(denses)
    
    assert rel_err(upsilon_mpo_to_dense(mpo_summed), dense_summed) < 1e-12

# --- MC Parity Tests ---

@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("dual_transform", ["id", "T", "conj", "dag"])
def test_mc_mpo_parity_discrete(k, dual_transform):
    sp = get_sim_params()
    H = small_hamiltonian()
    
    ups_dense, _ = run_mc_upsilon(
        operator=H, sim_params=sp, timesteps=[0.1]*k,
        num_sequences=4, ensemble="discrete", dual_transform=dual_transform, seed=42
    )
    ups_mpo, _ = run_mc_upsilon_mpo(
        operator=H, sim_params=sp, timesteps=[0.1]*k,
        num_sequences=4, ensemble="discrete", dual_transform=dual_transform, seed=42,
        compress_every=10000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    ups_mpo_dense = upsilon_mpo_to_dense(ups_mpo)
    assert rel_err(ups_dense, ups_mpo_dense) < 1e-12

@pytest.mark.parametrize("sampling", ["uniform", "candidate_local"])
def test_mc_mpo_parity_continuous(sampling):
    sp = get_sim_params()
    H = small_hamiltonian()
    
    kwargs = {
        "operator": H, "sim_params": sp, "timesteps": [0.1, 0.1],
        "ensemble": "continuous", "sampling": sampling, "seed": 42
    }
    if sampling == "candidate_local":
        kwargs["num_candidates"] = 4
        kwargs["num_sequences"] = 8
    else:
        kwargs["num_sequences"] = 8

    ups_dense, _ = run_mc_upsilon(**kwargs)
    ups_mpo, _ = run_mc_upsilon_mpo(
        compress_every=10000, tol=0.0, max_bond_dim=None, n_sweeps=0, **kwargs
    )
    assert rel_err(ups_dense, upsilon_mpo_to_dense(ups_mpo)) < 1e-12

def test_mc_mpo_parity_noisy():
    sp = get_sim_params()
    sp.noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.05}])
    H = small_hamiltonian()
    
    ups_dense, _ = run_mc_upsilon(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1], num_sequences=1, seed=42
    )
    ups_mpo, _ = run_mc_upsilon_mpo(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1], num_sequences=1, seed=42,
        compress_every=10000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    assert rel_err(ups_dense, upsilon_mpo_to_dense(ups_mpo)) < 1e-10

# --- SIS Parity Tests ---

@pytest.mark.parametrize("proposal", ["uniform", "local", "mixture"])
@pytest.mark.parametrize("dual_transform", ["id", "T"])
def test_sis_mpo_parity(proposal, dual_transform):
    sp = get_sim_params()
    H = small_hamiltonian()
    
    ups_dense, _ = run_sis_upsilon(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1],
        num_particles=8, proposal=proposal, resample=False, rejuvenate=False, dual_transform=dual_transform, seed=123
    )
    ups_mpo, _ = run_sis_upsilon_mpo(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1],
        num_particles=8, proposal=proposal, resample=False, rejuvenate=False, dual_transform=dual_transform, seed=123,
        compress_every=10000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    assert rel_err(ups_dense, upsilon_mpo_to_dense(ups_mpo)) < 1e-12

# --- Compression Tests ---

def test_compressed_mc_smoke():
    sp = get_sim_params()
    H = small_hamiltonian()
    
    ups_dense, _ = run_mc_upsilon(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        num_sequences=16, seed=42
    )
    ups_mpo, meta = run_mc_upsilon_mpo(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        num_sequences=16, seed=42,
        compress_every=4, tol=1e-10, max_bond_dim=32, n_sweeps=1
    )
    ups_mpo_dense = upsilon_mpo_to_dense(ups_mpo)
    
    assert "bond_dim_final" in meta
    assert ups_mpo_dense.shape == ups_dense.shape
    assert rel_err(ups_dense, ups_mpo_dense) < 1e-6

def test_compression_delta():
    sp = get_sim_params()
    H = small_hamiltonian()
    
    ups_uncompressed, _ = run_mc_upsilon_mpo(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        num_sequences=16, seed=777,
        compress_every=1000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    ups_compressed, _ = run_mc_upsilon_mpo(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        num_sequences=16, seed=777,
        compress_every=2, tol=1e-12, max_bond_dim=None, n_sweeps=1
    )
    
    d_u = upsilon_mpo_to_dense(ups_uncompressed)
    d_c = upsilon_mpo_to_dense(ups_compressed)
    assert rel_err(d_u, d_c) < 1e-10

def test_monotonic_compression():
    sp = get_sim_params()
    H = small_hamiltonian()
    
    ups_strong, _ = run_mc_upsilon_mpo(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        num_sequences=16, seed=888,
        compress_every=2, tol=1e-4, max_bond_dim=8, n_sweeps=1
    )
    ups_weak, _ = run_mc_upsilon_mpo(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        num_sequences=16, seed=888,
        compress_every=2, tol=1e-10, max_bond_dim=64, n_sweeps=1
    )
    ups_exact, _ = run_mc_upsilon_mpo(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        num_sequences=16, seed=888,
        compress_every=1000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    
    err_strong = rel_err(upsilon_mpo_to_dense(ups_strong), upsilon_mpo_to_dense(ups_exact))
    err_weak = rel_err(upsilon_mpo_to_dense(ups_weak), upsilon_mpo_to_dense(ups_exact))
    
    assert err_weak < err_strong + 1e-12

# --- Metadata and Failure modes ---

def test_metadata_and_failure_modes():
    sp = get_sim_params()
    H = small_hamiltonian()
    
    _, meta = run_mc_upsilon_mpo(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1],
        num_sequences=4, seed=42
    )
    
    assert "bond_dim_final" in meta
    assert "max_bond_final" in meta
    assert "compression_tol" in meta
    assert "compression_max_bond_dim" in meta
    assert "compression_n_sweeps" in meta
    
    with pytest.raises(NotImplementedError):
        run_sis_upsilon_mpo(
            operator=H, sim_params=sp, timesteps=[0.1, 0.1],
            num_particles=4, rejuvenate=True
        )

# --- Small Multi-Qubit Parity ---

@pytest.mark.slow
def test_small_multi_qubit_parity():
    sp = get_sim_params()
    # N=2 hamiltonian
    H = MPO.hamiltonian(length=2, one_body=[(1.0, "X")], two_body=[])
    
    ups_dense, _ = run_mc_upsilon(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1],
        num_sequences=4, seed=999
    )
    ups_mpo, _ = run_mc_upsilon_mpo(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1],
        num_sequences=4, seed=999,
        compress_every=1000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    
    assert rel_err(ups_dense, upsilon_mpo_to_dense(ups_mpo)) < 1e-10