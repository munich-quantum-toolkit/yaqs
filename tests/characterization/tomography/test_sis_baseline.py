import numpy as np
import pytest
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.characterization.tomography import run

def get_matrix(res):
    """Robustly extract dense Choi matrix from ProcessTensor or ndarray."""
    if hasattr(res, "dense_choi") and res.dense_choi is not None:
        return res.dense_choi
    if hasattr(res, "reconstruct_comb_choi"):
        return res.reconstruct_comb_choi()
    return res

def test_sis_mc_baseline_consistency():
    """Verify that SIS and MC converge similarly on a tiny deterministic case."""
    # L=1 system, k=1 
    op = MPO.hamiltonian(length=1, one_body=[(1.0, "X")])
    params = AnalogSimParams(dt=0.1, solver="MCWF", get_state=True)
    timesteps = [0.1]
    
    # Reference
    res_ex = run(op, params, timesteps, method="exhaustive", output="dense")
    ex_mat = get_matrix(res_ex)
    
    # MC
    res_mc = run(op, params, timesteps, method="mc", output="dense", num_samples=512, seed=42, num_trajectories=1)
    mc_mat = get_matrix(res_mc)
    
    # SIS
    res_sis = run(op, params, timesteps, method="sis", output="dense", num_samples=512, seed=42, num_trajectories=1)
    sis_mat = get_matrix(res_sis)
    
    # Check that both are decently close to exhaustive for N=512
    err_mc = np.linalg.norm(mc_mat - ex_mat)
    err_sis = np.linalg.norm(sis_mat - ex_mat)
    
    # Error should be reasonably small for N=512
    assert err_mc < 1.0
    assert err_sis < 1.0
    
    # Optional: ensure they are within the same order of magnitude
    assert abs(err_mc - err_sis) < 0.5

def test_sis_mc_matching_accuracy():
    """Strictly verify that SIS and MC reach similar accuracy levels for same N."""
    op = MPO.hamiltonian(length=1, one_body=[(1.0, "X")])
    params = AnalogSimParams(dt=0.1, solver="MCWF")
    timesteps = [0.2]
    
    res_ex = run(op, params, timesteps, method="exhaustive", output="dense")
    ex_mat = get_matrix(res_ex)
    
    N = 1000
    res_mc = run(op, params, timesteps, method="mc", num_samples=N, seed=42, output="dense", num_trajectories=1)
    res_sis = run(op, params, timesteps, method="sis", num_samples=N, seed=42, output="dense", num_trajectories=1)
    
    err_mc = np.linalg.norm(get_matrix(res_mc) - ex_mat)
    err_sis = np.linalg.norm(get_matrix(res_sis) - ex_mat)
    
    # At N=1000, they should be very comparable
    assert abs(err_mc - err_sis) < 0.2
    assert err_mc < 0.5
    assert err_sis < 0.5

def test_sis_dense_mpo_consistency():
    """Verify that SIS SamplingData produces matching dense and MPO reconstructions."""
    op = MPO.hamiltonian(length=1, one_body=[(1.0, "X")])
    params = AnalogSimParams(dt=0.1, solver="TJM", get_state=True)
    timesteps = [0.1]
    
    # Get SamplingData (indirectly via output)
    # We'll just run them and compare
    res_dense = run(op, params, timesteps, method="sis", output="dense", num_samples=50, seed=1, num_trajectories=1)
    res_mpo = run(op, params, timesteps, method="sis", output="mpo", num_samples=50, seed=1, num_trajectories=1)
    
    mpo_mat = res_mpo.to_matrix()
    dense_mat = get_matrix(res_dense)
    
    np.testing.assert_allclose(dense_mat, mpo_mat, atol=1e-12)

def test_sis_unsupported_proposals():
    """Verify that baseline SIS raises ValueError for non-uniform proposals."""
    op = MPO.hamiltonian(length=1, one_body=[(1.0, "X")])
    params = AnalogSimParams(dt=0.1, solver="MCWF")
    
    with pytest.raises(ValueError, match="not currently supported"):
        # We need to bypass the Literal static check if using type checkers, 
        # but in raw python we just pass the string.
        run(op, params, method="sis", proposal="local", num_trajectories=1) # type: ignore

def test_sis_convergence():
    """Verify that SIS error decreases when increasing num_samples."""
    op = MPO.hamiltonian(length=1, one_body=[(1.0, "X")])
    params = AnalogSimParams(dt=0.1, solver="MCWF")
    timesteps = [0.2]
    
    res_ex = run(op, params, timesteps, method="exhaustive", output="dense")
    ex_mat = get_matrix(res_ex)
    
    errors = []
    for n in [32, 128, 512]:
        res = run(op, params, timesteps, method="sis", num_samples=n, seed=42, output="dense", num_trajectories=1)
        errors.append(np.linalg.norm(get_matrix(res) - ex_mat))
    
    # Error should generally decrease (though sampling noise exists)
    # For this tiny k=1 case, it should be fairly stable
    assert errors[-1] < errors[0]
    
    # Error should generally decrease (though sampling noise exists)
    # For this tiny k=1 case, it should be fairly stable
    assert errors[-1] < errors[0]
