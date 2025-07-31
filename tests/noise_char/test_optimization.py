import numpy as np
import pytest
import os
import shutil

from mqt.yaqs.noise_char import optimization

def test_trapezoidal_basic():
    x = np.linspace(0, 1, 5)
    y = x**2
    result = optimization.trapezoidal(y, x)
    # The last value should be close to the integral of x^2 from 0 to 1, which is 1/3
    assert np.isclose(result[-1], 1/3, atol=1e-2)

def test_trapezoidal_shape_and_error():
    x = np.linspace(0, 1, 10)
    y = np.sin(x)
    result = optimization.trapezoidal(y, x)
    assert result.shape == y.shape
    # Should raise ValueError if lengths mismatch
    with pytest.raises(ValueError):
        optimization.trapezoidal(y, x[:-1])

def test_loss_class_history_and_reset(tmp_path):
    class DummyLoss(optimization.loss_class):
        def __init__(self):
            self.d = 2
            self.print_to_file = False
            self.exp_vals_traj = np.zeros((3, 2, 5))
            self.t = np.arange(5)
            self.work_dir = str(tmp_path)
    loss = DummyLoss()
    loss.reset()
    assert loss.n_eval == 0
    assert loss.x_history == []
    loss.set_history([[1,2]], [3], [[1,2]], [0.1])
    assert loss.n_eval == 1
    assert loss.x_history == [[1,2]]
    assert loss.f_history == [3]
    assert loss.diff_avg_history == [0.1]

def test_loss_class_compute_avg_and_diff():
    class DummyLoss(optimization.loss_class):
        def __init__(self):
            self.n_avg = 2
            self.x_history = [np.array([1,2]), np.array([2,3]), np.array([3,4])]
            self.x_avg_history = []
            self.diff_avg_history = []
    loss = DummyLoss()
    loss.compute_avg()
    assert np.allclose(loss.x_avg_history[-1], np.mean(loss.x_history[2:], axis=0))
    loss.x_avg_history.append(np.array([2,3]))
    loss.compute_diff_avg()
    assert loss.diff_avg_history[-1] == np.max(np.abs(loss.x_avg_history[-1] - loss.x_avg_history[-2]))

def test_loss_class_write_opt_traj(tmp_path):
    n_obs_site = 3
    n_jump_sites = 2
    L = 2
    n_t = 5

    class DummyLoss(optimization.loss_class):
        def __init__(self):
            self.d = 2
            self.print_to_file = False
            self.ref_traj = np.ones((n_obs_site, L, n_t))
            self.t = np.arange(n_t)
            self.work_dir = str(tmp_path)
            self.n_eval = 1

    loss = DummyLoss()
    loss.write_opt_traj()
    out_file = os.path.join(loss.work_dir, "opt_traj_1.txt")
    assert os.path.exists(out_file)
    data = np.loadtxt(out_file)
    assert data.shape[1] == 1 + n_obs_site*n_jump_sites  # t + 3 obs * 2 sites

def test_loss_class_set_file_name_and_write_to_file(tmp_path):
    class DummyLoss(optimization.loss_class):
        def __init__(self):
            self.d = 2
            self.print_to_file = True
            self.work_dir = str(tmp_path)
    loss = DummyLoss()
    file_name = os.path.join(loss.work_dir, "testfile")
    loss.set_file_name(file_name, reset=True)
    assert os.path.exists(file_name + ".txt")
    loss.n_eval = 1
    loss.write_to_file(file_name + ".txt", 0.5, np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    with open(file_name + ".txt") as f:
        lines = f.readlines()
    assert any("1" in line for line in lines)

def test_loss_class_2d_call(monkeypatch):

    n_obs_site = 3
    n_jump_sites = 2
    L = 2
    n_t = 5

    class DummySimParams:
        def set_gammas(self, a, b): pass
    def dummy_traj_der(sim_params):
        t = np.arange(n_t)
        exp_vals_traj = np.ones((n_obs_site, L, n_t))
        d_On_d_gk = np.ones((n_jump_sites, n_obs_site, L, n_t))
        avg_min_max_traj_time = [None, None, None]
        return t, exp_vals_traj, d_On_d_gk, avg_min_max_traj_time
    
    sim_params = DummySimParams()
    ref_traj = np.ones((n_obs_site, L, n_t))
    loss = optimization.loss_class_2d(sim_params, ref_traj, dummy_traj_der)
    x = np.array([0.5, 0.5])
    f, grad, sim_time, avg = loss(x)
    assert isinstance(f, float)
    assert grad.shape == (1,)
    assert isinstance(sim_time, float)
    assert isinstance(avg, list)


def test_loss_class_nd_call(monkeypatch):
    n_obs_site = 3
    n_jump_sites = 2
    L = 2
    n_t = 5

    class DummySimParams:
        def __init__(self):
            self.gamma_rel = [0.1, 0.2]
            self.gamma_deph = [0.3, 0.4]
        def set_gammas(self, rel, deph): pass
    def dummy_traj_der(sim_params):
        t = np.arange(n_t)
        exp_vals_traj = np.ones((n_obs_site, L, n_t))
        d_On_d_gk = np.ones((n_jump_sites, n_obs_site, L, n_t))
        avg_min_max_traj_time = [1,2,3]
        return t, exp_vals_traj, d_On_d_gk, avg_min_max_traj_time
    sim_params = DummySimParams()
    ref_traj = np.ones((n_obs_site, L, n_t))
    loss = optimization.loss_class_nd(sim_params, ref_traj, dummy_traj_der)
    x = np.array([0.5]*L*n_jump_sites)
    f, grad, sim_time, avg = loss(x)
    assert isinstance(f, float)
    assert grad.shape == (4,)
    assert isinstance(sim_time, float)
    assert isinstance(avg, list)

def test_adam_optimizer_runs(tmp_path):
    class DummyLoss(optimization.loss_class):
        def __init__(self):
            self.d = 2
            self.print_to_file = False
            self.work_dir = str(tmp_path)
            self.f_history = []
            self.x_history = []
            self.x_avg_history = []
            self.diff_avg_history = []
            self.t = np.arange(3)
            self.exp_vals_traj = np.ones((3,2,3))
        def __call__(self, x):
            return 0.1, np.array([0.01, 0.01]), 0.01, [None, None, None]
    loss = DummyLoss()
    x0 = np.array([0.5, 0.5])
    f_hist, x_hist, x_avg_hist, t, exp_vals_traj = optimization.adam_optimizer(loss, x0, max_iterations=3)
    assert isinstance(f_hist, list)
    assert isinstance(x_hist, list)
    assert isinstance(x_avg_hist, list)
    assert isinstance(t, np.ndarray)
    assert isinstance(exp_vals_traj, np.ndarray)