# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for the optimization module's noise characterization functionality."""

from __future__ import annotations

import contextlib
import pathlib
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs.noise_char import optimization, propagation
from mqt.yaqs.noise_char.optimization import LossClass
from mqt.yaqs.noise_char.propagation import SimulationParameters

if TYPE_CHECKING:
    from collections.abc import Generator


def test_trapezoidal_basic() -> None:
    """Test the basic functionality of the trapezoidal integration method.

    This test verifies that the `optimization.trapezoidal` function correctly computes
    the cumulative integral of y = x^2 over the interval [0, 1] using 5 sample points.
    It asserts that the final value of the cumulative integral is close to the analytical
    result of the definite integral of x^2 from 0 to 1, which is 1/3, within a tolerance of 1e-2.
    """
    x = np.linspace(0, 1, 20)
    y = x**2
    result = optimization.trapezoidal(y, x)
    # The last value should be close to the integral of x^2 from 0 to 1, which is 1/3
    assert np.isclose(result[-1], 1 / 3, atol=1e-2)


def test_trapezoidal_shape_and_error() -> None:
    """Test the `trapezoidal` function from the `optimization` module for correct output shape and error handling.

    This test verifies that:
    - The output of `optimization.trapezoidal(y, x)` has the same shape as the input array `y`.
    - A `ValueError` is raised when the lengths of `y` and `x` do not match.
    """
    x = np.linspace(0, 1, 10)
    y = np.sin(x)
    result = optimization.trapezoidal(y, x)
    assert result.shape == y.shape
    # Should raise ValueError if lengths mismatch
    with pytest.raises(ValueError, match="Mismatch in the number of elements between x and y"):
        optimization.trapezoidal(y, x[:-1])


def arrays_equal(list1: list[np.ndarray], list2: list[np.ndarray]) -> bool:
    """Check if two lists of NumPy arrays are equal element-wise.

    Compares two lists of numpy arrays by:
    - Checking if they have the same length.
    - Verifying that each corresponding pair of arrays is equal using `np.array_equal`.

    Parameters
    ----------
    list1 : List[np.ndarray]
        The first list of numpy arrays to compare.
    list2 : List[np.ndarray]
        The second list of numpy arrays to compare.

    Returns:
    -------
    bool
        True if both lists have the same length and all corresponding arrays are equal,
        False otherwise.
    """
    if len(list1) != len(list2):
        return False
    return all(map(np.array_equal, list1, list2))


def test_loss_class_history_and_reset(tmp_path: pathlib.Path) -> None:
    """Tests the history management and reset functionality of a loss class.

    This test defines a DummyLoss class inheriting from `optimization.LossClass` and verifies:
    - The `reset()` method correctly initializes evaluation count and history attributes.
    - The `set_history()` method updates the history attributes (`x_history`, `f_history`, `
    diff_avg_history`) and evaluation count (`n_eval`) as expected.

    Args:
        tmp_path: pytest fixture providing a temporary directory for file operations.
    Asserts:
        - After reset, `n_eval` is 0 and `x_history` is an empty list.
        - After setting history, `n_eval` is incremented, and history attributes match the provided values.
    """

    class DummyLoss(optimization.LossClass):
        def __init__(self) -> None:
            self.d = 2
            self.print_to_file = False
            self.exp_vals_traj = np.zeros((3, 2, 5))
            self.t = np.arange(5)
            self.work_dir = tmp_path

    loss = DummyLoss()
    loss.reset()
    assert loss.n_eval == 0
    assert loss.x_history == []
    loss.set_history([np.array([1, 2])], [3], [np.array([1, 2])], [0.1])
    assert loss.n_eval == 1
    assert arrays_equal(loss.x_history, [np.array([1, 2])])
    assert loss.f_history == [3]
    assert loss.diff_avg_history == [0.1]


def test_loss_class_compute_avg_and_diff() -> None:
    """Test the computation of average and difference in the loss class.

    This test defines a dummy subclass of `optimization.LossClass` with preset history values.
    It verifies that:
    - The `compute_avg()` method correctly computes and appends
    the average of the last `n_avg` entries in `x_history` to `x_avg_history`.
    - The `compute_diff_avg()` method correctly computes and appends the maximum
    absolute difference between the last two averages in `x_avg_history` to `diff_avg_history`.
    Assertions:
        - The last entry in `x_avg_history` matches the mean of the last `n_avg` entries in `x_history`.
        - The last entry in `diff_avg_history` matches the maximum absolute difference
        between the last two averages in `x_avg_history`.
    """

    class DummyLoss(optimization.LossClass):
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


def test_write_opt_traj_creates_correct_file_and_content() -> None:
    """Test for the LossClass.write_opt_traj method.

    The test verifies that:
    - The output file `opt_traj_{n_eval}.txt` is created in the specified work directory.
    - The header line starts with 't' and contains correct observable labels formatted as
      'x0', 'y0', 'z0', etc., matching the dimensions of the `exp_vals_traj`.
    - The number of data lines matches the number of time points.
    - Each data line contains the correct number of columns: one for time plus one for each observable/site combination.
    - The time values in the file match the original `t` array within a numerical tolerance.

    This ensures that the trajectory data is saved properly for later analysis or visualization.
    """
    # Create an instance
    loss = LossClass()

    # Setup dimension parameters
    n_obs_site = 2  # e.g. x and y observables
    sites = 3  # number of sites
    n_t = 5  # number of time points
    loss.d = sites  # just to set d consistent for other methods if needed

    # Setup the exp_vals_traj with shape (n_obs_site, sites, n_t)
    rng = np.random.default_rng()
    loss.exp_vals_traj = rng.random((n_obs_site, sites, n_t))

    # Setup the time vector
    loss.t = np.linspace(0, 1, n_t)

    # Use a temporary directory as work_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        loss.work_dir = Path(tmpdir)
        loss.n_eval = 7  # evaluation index for filename

        # Call the method
        loss.write_opt_traj()

        # Check that the file was created
        output_file = loss.work_dir / f"opt_traj_{loss.n_eval}.txt"
        assert output_file.exists()

        # Read the file content using Path.read_text()
        file_content = output_file.read_text(encoding="utf-8")
        lines = file_content.splitlines(keepends=True)

        # The first line should be the header starting with 't'
        header = lines[0].strip()
        expected_header_prefix = "# t"
        assert header.startswith(expected_header_prefix)

        # Check header contains correct observable labels like x0, y0 (depending on n_obs_site and sites)
        # Construct expected header pattern
        expected_labels = [obs + str(i) for obs in ["x", "y", "z"][:n_obs_site] for i in range(sites)]
        for label in expected_labels:
            assert label in header

        # The data lines count should match number of time points
        data_lines = lines[1:]
        assert len(data_lines) == n_t

        # Check that each data line has expected number of columns: 1 (time) + n_obs_site * sites
        expected_cols = 1 + n_obs_site * sites
        for line in data_lines:
            cols = re.split(r"\s+", line.strip())
            assert len(cols) == expected_cols

        # Check the time column matches the time vector in file (within precision)
        file_times = [float(line.split()[0]) for line in data_lines]
        np.testing.assert_allclose(file_times, loss.t, rtol=1e-6)


def test_loss_class_set_work_dir(tmp_path: pathlib.Path) -> None:
    """Tests the functionality of setting a file name and writing to a file in a custom loss class.

    This test defines a DummyLoss class inheriting from `optimization.LossClass`, sets up a working directory,
    and verifies that:
    - The file name is correctly set and the file is created.

    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest for file operations.
    Asserts:
        - The file with the expected name exists after calling `set_work_dir`.
    """

    class DummyLoss(optimization.LossClass):
        def __init__(self) -> None:
            self.d = 2
            self.print_to_file = True

    loss = DummyLoss()

    loss_file = tmp_path / "loss_x_history.txt"
    loss_file_avg = tmp_path / "loss_x_history_avg.txt"

    loss.set_work_dir(tmp_path, reset=True)
    assert loss.history_file_name == loss_file
    assert loss.history_avg_file_name == loss_file_avg


@pytest.fixture
def setup_loss_class_files() -> Generator[Path, None, None]:
    """Pytest fixture that provides a temporary directory as a Path object for use in tests.

    The temporary directory is created before the test runs and automatically cleaned up
    after the test finishes, ensuring no leftover files or directories.

    Yields:
    -------
    Path
        A Path object pointing to the temporary directory for test file operations.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        yield base_path


def test_write_to_file_creates_and_appends(setup_loss_class_files: Path) -> None:
    """Test for  the `write_to_file` method.

    This test verifies:
    - The history file is created with the expected header line.
    - The data line written matches the provided function value, parameters, and gradient,
      formatted with 6 decimal places.
    - The average history file is also created with the expected header.

    Parameters:
    -----------
    setup_loss_class_files : pytest fixture
        Provides a temporary directory path used for file operations during testing.

    Procedure:
    ----------
    - Initialize `LossClass` with `print_to_file = True` and set necessary attributes.
    - Provide sample function value, parameter vector, and gradient.
    - Call `write_to_file` to write to the history file.
    - Read and assert the contents of the history and average history files.

    Expected outcome:
    -----------------
    - Both history and average history files exist.
    - Files contain correctly formatted headers and data lines.
    """
    base_path = setup_loss_class_files

    # Instantiate LossClass and configure attributes
    loss_obj = LossClass()
    loss_obj.print_to_file = True
    loss_obj.n_eval = 1
    loss_obj.d = 3  # dimensionality

    # Set the file paths inside the temp directory
    loss_obj.history_file_name = base_path / "loss_x_history.txt"
    loss_obj.history_avg_file_name = base_path / "loss_x_history_avg.txt"

    # Data for the test
    f = 1.2345
    x = np.array([0.1, 0.2, 0.3])
    grad = np.array([0.01, 0.02, 0.03])

    # Call the method with the main output file (can be history_file_name for example)
    loss_obj.write_to_file(loss_obj.history_file_name, f, x, grad)

    # Read all lines from history file using Path.read_text()
    history_text = loss_obj.history_file_name.read_text(encoding="utf-8")
    lines = history_text.splitlines(keepends=True)

    # First line: header
    expected_header = "# iter  loss  x1  x2  x3    grad_x1  grad_x2  grad_x3\n"
    assert lines[0] == expected_header

    # Second line: data line, should contain n_eval, f, x, grad formatted
    expected_line = f"{loss_obj.n_eval}    {f}  0.100000  0.200000  0.300000    0.010000  0.020000  0.030000\n"
    assert lines[1] == expected_line

    # Read first line from average history file using Path.read_text()
    avg_text = loss_obj.history_avg_file_name.read_text(encoding="utf-8")
    avg_header = avg_text.splitlines(keepends=True)[0]

    expected_avg_header = "# iter  loss  x1_avg  x2_avg  x3_avg    grad_x1  grad_x2  grad_x3\n"
    assert avg_header == expected_avg_header


def test_write_to_file_does_nothing_when_disabled(setup_loss_class_files: Path) -> None:
    """Test for the `write_to_file` method.

    This ensures that disabling file output correctly prevents any file I/O operations.

    Parameters:
    -----------
    setup_loss_class_files : pytest fixture
        Provides a temporary directory path used for file operations during testing.

    Steps:
    ------
    - Initializes a `LossClass` instance with `print_to_file` set to False.
    - Sets up file paths and other required attributes.
    - Calls `write_to_file`.
    - Asserts that no output files are created.

    Expected outcome:
    -----------------
    - The history files should not exist after calling `write_to_file` with output disabled.
    """
    base_path = setup_loss_class_files

    loss_obj = LossClass()
    loss_obj.print_to_file = False  # Writing disabled
    loss_obj.n_eval = 1
    loss_obj.d = 2
    loss_obj.history_file_name = base_path / "loss_x_history.txt"
    loss_obj.history_avg_file_name = base_path / "loss_x_history_avg.txt"

    f = 0.5
    x = np.array([0.1, 0.2])
    grad = np.array([0.01, 0.02])

    loss_obj.write_to_file(loss_obj.history_file_name, f, x, grad)

    # Files should not exist because print_to_file=False
    assert not loss_obj.history_file_name.exists()
    assert not loss_obj.history_avg_file_name.exists()


def test_loss_class_2_call() -> None:
    """Unit test for the `optimization.LossClass2` function.

    This test verifies that the loss function returned by `LossClass2`:
    - Accepts a parameter vector `x` and returns a tuple `(f, grad, sim_time, avg)`.
    - Ensures the output `f` is a float.
    - Ensures the output `grad` is a numpy array of shape (1,).
    - Ensures the output `sim_time` is a float.
    - Ensures the output `avg` is a list.
    The test uses dummy simulation parameters and a dummy trajectory derivative function
    to mock the behavior of the actual simulation.
    """
    n_obs_site = 3
    n_jump_sites = 2
    sites = 2
    n_t = 5

    def dummy_traj_der(sim_params: SimulationParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[None]]:
        t = np.arange(n_t)
        exp_vals_traj = np.ones((n_obs_site, sim_params.L, n_t))
        d_on_d_gk = np.ones((n_jump_sites, n_obs_site, sim_params.L, n_t))
        avg_min_max_traj_time = [None, None, None]
        return t, exp_vals_traj, d_on_d_gk, avg_min_max_traj_time

    sim_params = SimulationParameters(sites, 0.1, 0.1)

    ref_traj = np.ones((n_obs_site, sites, n_t))
    loss = optimization.LossClass2(sim_params, ref_traj, dummy_traj_der)
    x = np.array([0.5, 0.5])
    f, grad, sim_time, avg = loss(x)
    assert isinstance(f, float)
    assert grad.shape == (2,)
    assert isinstance(sim_time, float)
    assert isinstance(avg, list)


def test_loss_class_2l_call() -> None:
    """Test the `LossClass2L` function from the `optimization` module.

    This test verifies that the loss function returned by `LossClass2L`:
    - Accepts a parameter vector `x` of appropriate shape.
    - Returns a tuple containing:
        - A scalar loss value (`f`).
        - A gradient array (`grad`) of expected shape.
        - A simulation time value (`sim_time`).
        - An average trajectory time list (`avg`).
    The test uses dummy simulation parameters and a dummy trajectory derivative function
    to mock the behavior of the underlying simulation. It asserts the types and shapes
    of the outputs to ensure correct functionality.

    Asserts:
        - The loss value is a float.
        - The gradient has the expected shape.
        - The simulation time is a float.
        - The average trajectory time is a list.
    """
    n_obs_site = 3
    n_jump_sites = 2
    sites = 2
    n_t = 5

    def dummy_traj_der(sim_params: SimulationParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[None]]:
        t = np.arange(n_t)
        exp_vals_traj = np.ones((n_obs_site, sim_params.L, n_t))
        d_on_d_gk = np.ones((n_jump_sites, n_obs_site, sim_params.L, n_t))
        avg_min_max_traj_time = [None, None, None]
        return t, exp_vals_traj, d_on_d_gk, avg_min_max_traj_time

    sim_params = SimulationParameters(sites, [0.1, 0.2], [0.3, 0.4])
    ref_traj = np.ones((n_obs_site, sites, n_t))
    loss = optimization.LossClass2L(sim_params, ref_traj, dummy_traj_der)
    x = np.array([0.5] * sites * n_jump_sites)
    f, grad, sim_time, avg = loss(x)
    assert isinstance(f, float)
    assert grad.shape == (4,)
    assert isinstance(sim_time, float)
    assert isinstance(avg, list)


def test_adam_optimizer_runs() -> None:
    """Test that the Adam optimizer runs successfully with a dummy loss function.

    This test defines a DummyLoss class inheriting from `optimization.LossClass` and
    verifies that the `adam_optimizer` function from the `optimization` module executes
    without errors for a small number of iterations. It checks that the returned values
    have the expected types.

    Asserts:
        - The function history (`f_hist`) is a list.
        - The parameter history (`x_hist`) is a list.
        - The averaged parameter history (`x_avg_hist`) is a list.
        - The time array (`t`) is a numpy ndarray.
        - The expectation values trajectory (`exp_vals_traj`) is a numpy ndarray.
    """
    sites = 2
    sim_time = 0.3
    dt = 0.1
    sim_params = SimulationParameters(sites=sites, gamma_rel=[0.1, 0.2], gamma_deph=[0.3, 0.4])
    sim_params.T = sim_time
    sim_params.dt = dt
    sim_params.N = 2

    t_list = np.arange(0, sim_time + dt, dt)

    ref_traj = np.ones((3, sites, len(t_list)))

    loss_function = optimization.LossClass2(sim_params, ref_traj, propagation.tjm_traj, print_to_file=False)

    x0 = np.array([0.5, 0.5])  # Initial guess for the parameters
    f_hist, x_hist, x_avg_hist, t, exp_vals_traj = optimization.adam_optimizer(loss_function, x0, max_iterations=3)
    assert isinstance(f_hist, list)
    assert isinstance(x_hist, list)
    assert isinstance(x_avg_hist, list)
    assert isinstance(t, np.ndarray)
    assert isinstance(exp_vals_traj, np.ndarray)

    for file_path in loss_function.work_dir.glob("opt_traj*.txt"):
        with contextlib.suppress(Exception):
            file_path.unlink()

    for file_path in loss_function.work_dir.glob("restart*.pkl"):
        with contextlib.suppress(Exception):
            file_path.unlink()

    for file_path in loss_function.work_dir.glob("performance*.txt"):
        with contextlib.suppress(Exception):
            file_path.unlink()
