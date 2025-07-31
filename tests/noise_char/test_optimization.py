from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from mqt.yaqs.noise_char import optimization
from mqt.yaqs.noise_char.propagation import SimulationParameters


def test_trapezoidal_basic() -> None:
    """Test the basic functionality of the trapezoidal integration method.
    This test verifies that the `optimization.trapezoidal` function correctly computes
    the cumulative integral of y = x^2 over the interval [0, 1] using 5 sample points.
    It asserts that the final value of the cumulative integral is close to the analytical
    result of the definite integral of x^2 from 0 to 1, which is 1/3, within a tolerance of 1e-2.
    """
    x = np.linspace(0, 1, 5)
    y = x**2
    result = optimization.trapezoidal(y, x)
    # The last value should be close to the integral of x^2 from 0 to 1, which is 1/3
    assert np.isclose(result[-1], 1 / 3, atol=1e-2)


def test_trapezoidal_shape_and_error() -> None:
    """Test the `trapezoidal` function from the `optimization` module for correct output shape and error handling.
    This test verifies that:
    - The output of `optimization.trapezoidal(y, x)` has the same shape as the input array `y`.
    - A `ValueError` is raised when the lengths of `y` and `x` do not match.

    Returns:
        None.
    """
    x = np.linspace(0, 1, 10)
    y = np.sin(x)
    result = optimization.trapezoidal(y, x)
    assert result.shape == y.shape
    # Should raise ValueError if lengths mismatch
    with pytest.raises(ValueError):
        optimization.trapezoidal(y, x[:-1])


def test_loss_class_history_and_reset(tmp_path: pathlib.Path) -> None:
    """Tests the history management and reset functionality of a loss class.
    This test defines a DummyLoss class inheriting from `optimization.loss_class` and verifies:
    - The `reset()` method correctly initializes evaluation count and history attributes.
    - The `set_history()` method updates the history attributes (`x_history`, `f_history`, `diff_avg_history`) and evaluation count (`n_eval`) as expected.

    Args:
        tmp_path: pytest fixture providing a temporary directory for file operations.
    Asserts:
        - After reset, `n_eval` is 0 and `x_history` is an empty list.
        - After setting history, `n_eval` is incremented, and history attributes match the provided values.
    """

    class DummyLoss(optimization.loss_class):
        def __init__(self) -> None:
            self.d = 2
            self.print_to_file = False
            self.exp_vals_traj = np.zeros((3, 2, 5))
            self.t = np.arange(5)
            self.work_dir = str(tmp_path)
    loss = DummyLoss()
    loss.reset()
    assert loss.n_eval == 0
    assert loss.x_history == []
    loss.set_history([[1, 2]], [3], [[1, 2]], [0.1])
    assert loss.n_eval == 1
    assert loss.x_history == [[1, 2]]
    assert loss.f_history == [3]
    assert loss.diff_avg_history == [0.1]


def test_loss_class_compute_avg_and_diff() -> None:
    """Test the computation of average and difference in the loss class.
    This test defines a dummy subclass of `optimization.loss_class` with preset history values.
    It verifies that:
    - The `compute_avg()` method correctly computes and appends the average of the last `n_avg` entries in `x_history` to `x_avg_history`.
    - The `compute_diff_avg()` method correctly computes and appends the maximum absolute difference between the last two averages in `x_avg_history` to `diff_avg_history`.
    Assertions:
        - The last entry in `x_avg_history` matches the mean of the last `n_avg` entries in `x_history`.
        - The last entry in `diff_avg_history` matches the maximum absolute difference between the last two averages in `x_avg_history`.
    """

    class DummyLoss(optimization.loss_class):
        def __init__(self) -> None:
            self.n_avg = 2
            self.x_history = [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])]
            self.x_avg_history = []
            self.diff_avg_history = []
    loss = DummyLoss()
    loss.compute_avg()
    assert np.allclose(loss.x_avg_history[-1], np.mean(loss.x_history[2:], axis=0))
    loss.x_avg_history.append(np.array([2, 3]))
    loss.compute_diff_avg()
    assert loss.diff_avg_history[-1] == np.max(np.abs(loss.x_avg_history[-1] - loss.x_avg_history[-2]))


def test_loss_class_write_opt_traj(tmp_path: pathlib.Path) -> None:
    """Test the `write_opt_traj` method of a loss class to ensure that the optimal trajectory
    is correctly written to a file.
    This test creates a dummy loss class with predefined trajectory data, invokes the
    `write_opt_traj` method, and verifies that the output file is created in the specified
    temporary directory. It then loads the data from the file and checks that the shape
    of the data matches the expected dimensions: one column for time and additional columns
    for each observable-site combination.

    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest for file output.
    Asserts:
        - The output file exists after writing.
        - The shape of the loaded data matches the expected number of columns.
    """
    n_obs_site = 3
    n_jump_sites = 2
    L = 2
    n_t = 5

    class DummyLoss(optimization.loss_class):
        def __init__(self) -> None:
            self.d = 2
            self.print_to_file = False
            self.ref_traj = np.ones((n_obs_site, L, n_t))
            self.t = np.arange(n_t)
            self.work_dir = str(tmp_path)
            self.n_eval = 1

    loss = DummyLoss()
    loss.write_opt_traj()
    out_file = os.path.join(loss.work_dir, "opt_traj_1.txt")
    assert pathlib.Path(out_file).exists()
    data = np.loadtxt(out_file)
    assert data.shape[1] == 1 + n_obs_site * n_jump_sites  # t + 3 obs * 2 sites


def test_loss_class_set_file_name_and_write_to_file(tmp_path: pathlib.Path) -> None:
    """Tests the functionality of setting a file name and writing to a file in a custom loss class.
    This test defines a DummyLoss class inheriting from `optimization.loss_class`, sets up a working directory,
    and verifies that:
    - The file name is correctly set and the file is created.
    - Data can be written to the file using the `write_to_file` method.
    - The written file contains the expected evaluation data.

    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest for file operations.
    Asserts:
        - The file with the expected name exists after calling `set_file_name`.
        - The file contains the expected evaluation data after calling `write_to_file`.
    """

    class DummyLoss(optimization.loss_class):
        def __init__(self) -> None:
            self.d = 2
            self.print_to_file = True
            self.work_dir = str(tmp_path)
    loss = DummyLoss()
    file_name = os.path.join(loss.work_dir, "testfile")
    loss.set_file_name(file_name, reset=True)
    assert pathlib.Path(file_name + ".txt").exists()
    loss.n_eval = 1
    loss.write_to_file(file_name + ".txt", 0.5, np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    with open(file_name + ".txt", encoding="utf-8") as f:
        lines = f.readlines()
    assert any("1" in line for line in lines)


def test_loss_class_2d_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit test for the `optimization.loss_class_2d` function.
    This test verifies that the loss function returned by `loss_class_2d`:
    - Accepts a parameter vector `x` and returns a tuple `(f, grad, sim_time, avg)`.
    - Ensures the output `f` is a float.
    - Ensures the output `grad` is a numpy array of shape (1,).
    - Ensures the output `sim_time` is a float.
    - Ensures the output `avg` is a list.
    The test uses dummy simulation parameters and a dummy trajectory derivative function to mock the behavior of the actual simulation.
    """
    n_obs_site = 3
    n_jump_sites = 2
    L = 2
    n_t = 5


    def dummy_traj_der(sim_params):
        t = np.arange(n_t)
        exp_vals_traj = np.ones((n_obs_site, L, n_t))
        d_On_d_gk = np.ones((n_jump_sites, n_obs_site, L, n_t))
        avg_min_max_traj_time = [None, None, None]
        return t, exp_vals_traj, d_On_d_gk, avg_min_max_traj_time

    sim_params = SimulationParameters(L, 0.1, 0.1)

    ref_traj = np.ones((n_obs_site, L, n_t))
    loss = optimization.loss_class_2d(sim_params, ref_traj, dummy_traj_der)
    x = np.array([0.5, 0.5])
    f, grad, sim_time, avg = loss(x)
    assert isinstance(f, float)
    assert grad.shape == (1,)
    assert isinstance(sim_time, float)
    assert isinstance(avg, list)


def test_loss_class_nd_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the `loss_class_nd` function from the `optimization` module.
    This test verifies that the loss function returned by `loss_class_nd`:
    - Accepts a parameter vector `x` of appropriate shape.
    - Returns a tuple containing:
        - A scalar loss value (`f`).
        - A gradient array (`grad`) of expected shape.
        - A simulation time value (`sim_time`).
        - An average trajectory time list (`avg`).
    The test uses dummy simulation parameters and a dummy trajectory derivative function
    to mock the behavior of the underlying simulation. It asserts the types and shapes
    of the outputs to ensure correct functionality.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.
    Asserts:
        - The loss value is a float.
        - The gradient has the expected shape.
        - The simulation time is a float.
        - The average trajectory time is a list.
    """
    n_obs_site = 3
    n_jump_sites = 2
    L = 2
    n_t = 5


    def dummy_traj_der(sim_params):
        t = np.arange(n_t)
        exp_vals_traj = np.ones((n_obs_site, L, n_t))
        d_On_d_gk = np.ones((n_jump_sites, n_obs_site, L, n_t))
        avg_min_max_traj_time = [1, 2, 3]
        return t, exp_vals_traj, d_On_d_gk, avg_min_max_traj_time
    
    sim_params = SimulationParameters(L,[0.1, 0.2], [0.3, 0.4])
    ref_traj = np.ones((n_obs_site, L, n_t))
    loss = optimization.loss_class_nd(sim_params, ref_traj, dummy_traj_der)
    x = np.array([0.5] * L * n_jump_sites)
    f, grad, sim_time, avg = loss(x)
    assert isinstance(f, float)
    assert grad.shape == (4,)
    assert isinstance(sim_time, float)
    assert isinstance(avg, list)


def test_adam_optimizer_runs(tmp_path: pathlib.Path) -> None:
    """Test that the Adam optimizer runs successfully with a dummy loss function.
    This test defines a DummyLoss class inheriting from `optimization.loss_class` and
    verifies that the `adam_optimizer` function from the `optimization` module executes
    without errors for a small number of iterations. It checks that the returned values
    have the expected types.

    Args:
        tmp_path (pathlib.Path): Temporary directory path provided by pytest for file operations.
    Asserts:
        - The function history (`f_hist`) is a list.
        - The parameter history (`x_hist`) is a list.
        - The averaged parameter history (`x_avg_hist`) is a list.
        - The time array (`t`) is a numpy ndarray.
        - The expectation values trajectory (`exp_vals_traj`) is a numpy ndarray.
    """

    class DummyLoss(optimization.loss_class):
        def __init__(self) -> None:
            self.d = 2
            self.print_to_file = False
            self.work_dir = str(tmp_path)
            self.f_history = []
            self.x_history = []
            self.x_avg_history = []
            self.diff_avg_history = []
            self.t = np.arange(3)
            self.exp_vals_traj = np.ones((3, 2, 3))

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
