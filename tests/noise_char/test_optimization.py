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
import pickle  # noqa: S403
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs.noise_char import optimization, propagation
from mqt.yaqs.noise_char.optimization import LossClass

from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z

from mqt.yaqs.core.data_structures.networks import MPO, MPS


if TYPE_CHECKING:
    from collections.abc import Generator

class Parameters:
    def __init__(self) -> None:
        self.sites = 1
        self.sim_time = 0.6
        self.dt = 0.2
        self.order = 1
        self.threshold = 1e-4
        self.ntraj = 1
        self.max_bond_dim = 4
        self.j = 1
        self.g = 0.5


        self.times = np.arange(0, self.sim_time + self.dt, self.dt)

        self.n_obs = self.sites * 3  # x, y, z for each site
        self.n_jump = self.sites * 2  # lowering and pauli_z for each site
        self.n_t = len(self.times)

        self.gamma_rel = 0.1
        self.gamma_deph = 0.15


        self.d = 2



def create_loss_instance(tmp_path: Path, test: Parameters) -> LossClass:
    """Helper function to create a LossClass instance for testing."""


    h_0 = MPO()
    h_0.init_ising(test.sites, test.j, test.g)


    # Define the initial state
    init_state = MPS(test.sites, state='zeros')


    obs_list = [Observable(X(), site) for site in range(test.sites)]  + [Observable(Y(), site) for site in range(test.sites)] + [Observable(Z(), site) for site in range(test.sites)]


    sim_params = AnalogSimParams(observables=obs_list, elapsed_time=test.sim_time, dt=test.dt, num_traj=test.ntraj, max_bond_dim=test.max_bond_dim, threshold=test.threshold, order=test.order, sample_timesteps=True)


    ref_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(test.sites)], "strength": test.gamma_rel}]+[{"name": "pauli_z", "sites": [i for i in range(test.sites)], "strength": test.gamma_deph}])



    propagator = propagation.PropagatorWithGradients(
            sim_params=sim_params,
            hamiltonian=h_0,
            compact_noise_model=ref_noise_model,
            init_state=init_state
        )
    
    propagator.set_observable_list(obs_list)
    propagator.run(ref_noise_model)

    loss = optimization.LossClass(
        ref_traj=propagator.obs_traj,
        traj_gradients=propagator,
        working_dir=tmp_path,
        print_to_file=False,
    )

    return loss


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

    with pytest.raises(ValueError, match="x or y is None"):
        optimization.trapezoidal(None, np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="x or y is None"):
        optimization.trapezoidal(np.array([0, 1, 2]), None)


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



    test = Parameters()
    loss = create_loss_instance(tmp_path, test)

    loss.n_eval = 5  # Set to non-zero to test reset
    loss.x_history = [np.array([0, 1])]

    loss.reset()
    assert loss.n_eval == 0
    assert loss.x_history == []
    loss.set_history([np.array([1, 2])], [3], [np.array([1, 2])], [0.1])
    assert loss.n_eval == 1
    assert arrays_equal(loss.x_history, [np.array([1, 2])])
    assert loss.f_history == [3]
    assert loss.diff_avg_history == [0.1]


def test_loss_class_compute_avg_and_diff(tmp_path: pathlib.Path) -> None:
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

    test = Parameters()
    loss = create_loss_instance(tmp_path, test)


    loss.n_avg = 2
    loss.x_history = [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])]
    loss.x_avg_history = []
    loss.diff_avg_history = []

    loss.compute_avg()
    assert np.allclose(loss.x_avg_history[-1], np.mean(loss.x_history[2:], axis=0))
    loss.x_avg_history.append(np.array([2, 3]))
    loss.compute_diff_avg()
    assert loss.diff_avg_history[-1] == np.max(np.abs(loss.x_avg_history[-1] - loss.x_avg_history[-2]))


def test_write_opt_traj_creates_correct_file_and_content(tmp_path: pathlib.Path) -> None:
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
    test = Parameters()
    loss = create_loss_instance(tmp_path, test)

    obs_traj = np.zeros([test.n_obs, test.n_t])


    loss.n_eval = 7  # evaluation index for filename

    output_file = loss.work_dir / f"opt_traj_{loss.n_eval}.txt"

    # Call the method
    loss.write_traj(obs_traj, output_file)

    assert output_file.exists()

    # Read the file content using Path.read_text()
    file_content = output_file.read_text(encoding="utf-8")
    lines = file_content.splitlines(keepends=True)

    # The first line should be the header starting with 't'
    header = lines[0].strip()
    expected_header_prefix = "# t"
    print("!!!!!!!Header is:  ",header)
    assert header.startswith(expected_header_prefix)

    # Check header contains correct observable labels like x0, y0 (depending on n_obs_site and sites)
    # Construct expected header pattern
    expected_labels = ["obs_" + str(i) for i in range(test.n_obs)]
    for label in expected_labels:
        assert label in header

    expected_data = np.concatenate([np.array([test.times]), obs_traj], axis=0).T

    loaded_data = np.genfromtxt(output_file, skip_header=1)

    # Check that the loaded data matches the expected data
    assert np.allclose(loaded_data, expected_data)



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

    loss = create_loss_instance(tmp_path, Parameters())

    loss_file = tmp_path / "loss_x_history.txt"
    loss_file_avg = tmp_path / "loss_x_history_avg.txt"

    loss.set_work_dir(path=tmp_path)
    assert loss.history_file_name == loss_file
    assert loss.history_avg_file_name == loss_file_avg



def test_write_to_file_creates_and_appends(tmp_path: Path) -> None:
    """Test for  the `write_to_file` method.

    This test verifies:
    - The history file is created with the expected header line.
    - The data line written matches the provided function value, parameters, and gradient,
      formatted with 6 decimal places.
    - The average history file is also created with the expected header.

    Parameters:
    -----------
    tmp_path : pytest fixture
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
    

    # Instantiate LossClass and configure attributes
    loss = create_loss_instance(tmp_path, Parameters())
    loss.print_to_file = True
    loss.n_eval = 1
    loss.d = 3  # dimensionality

    # Set the file paths inside the temp directory
    loss.history_file_name = tmp_path / "loss_x_history.txt"
    loss.history_avg_file_name = tmp_path / "loss_x_history_avg.txt"

    # Data for the test
    f = 1.2345
    x = np.array([0.1, 0.2, 0.3])
    grad = np.array([0.01, 0.02, 0.03])

    # Call the method with the main output file (can be history_file_name for example)
    loss.write_to_file(loss.history_file_name, f, x, grad)


    # Read all lines from history file using Path.read_text()
    history_text = loss.history_file_name.read_text(encoding="utf-8")
    lines = history_text.splitlines(keepends=True)

    # First line: header
    expected_header = "# iter  loss  x1  x2  x3    grad_x1  grad_x2  grad_x3\n"
    assert lines[0] == expected_header

    # Second line: data line, should contain n_eval, f, x, grad formatted
    expected_line = f"{loss.n_eval}    {f}  0.100000  0.200000  0.300000    0.010000  0.020000  0.030000\n"
    assert lines[1] == expected_line


def test_write_to_file_does_nothing_when_disabled(tmp_path: Path) -> None:
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
    loss = create_loss_instance(tmp_path, Parameters())

    loss.print_to_file = False  # Writing disabled
    loss.n_eval = 1
    loss.d = 2
    loss.history_file_name = tmp_path / "loss_x_history.txt"
    loss.history_avg_file_name = tmp_path / "loss_x_history_avg.txt"

    f = 0.5
    x = np.array([0.1, 0.2])
    grad = np.array([0.01, 0.02])

    loss.write_to_file(loss.history_file_name, f, x, grad)

    # Files should not exist because print_to_file=False
    assert not loss.history_file_name.exists()
    assert not loss.history_avg_file_name.exists()



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
    test= Parameters()
    loss_function = create_loss_instance(pathlib.Path(tempfile.mkdtemp()), test)

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


def assert_list_of_arrays_equal(list1: list[np.ndarray], list2: list[np.ndarray]) -> None:
    """Assert that two sequences of NumPy arrays are exactly equal element-wise.

    This function checks:
      1. That both lists have the same length.
      2. That each corresponding array is exactly equal (no tolerance).

    Args:
        list1 (Sequence[np.ndarray]): The first list (or sequence) of arrays.
        list2 (Sequence[np.ndarray]): The second list (or sequence) of arrays.

    """
    assert len(list1) == len(list2), f"Length mismatch: {len(list1)} != {len(list2)}"
    for i, (arr1, arr2) in enumerate(zip(list1, list2)):
        np.testing.assert_array_equal(arr1, arr2, err_msg=f"Arrays differ at index {i}")


def test_restart_loads_x_m_v(tmp_path: Path) -> None:
    """Test that adam_optimizer correctly loads x, m, and v from a restart file.

    This test:
      1. Creates a mock `.pkl` restart file with known values for x, m, v, and histories.
      2. Calls adam_optimizer with restart=True.
      3. Asserts that the optimizer's loaded values match what was saved.
      4. Asserts if ValueError is raised when not existing restart file.

    Args:
        tmp_path (Path): Temporary path fixture provided by pytest.
    """

    test = Parameters()


    # Arrange: create mock restart file
    iteration = 5
    restart_file = tmp_path / f"restart_step_000{iteration}.pkl"

    x_history = [np.array([0.2]*test.d)] * iteration
    x_avg_history = [np.array([0.4]*test.d)] * iteration
    diff_avg_history = [2] * iteration
    f_history = [0] * iteration

    expected_x = x_history[-1]
    expected_m = np.array([0.01]*test.d)
    expected_v = np.array([0.02]*test.d)




    obs_traj = np.zeros([test.n_obs, test.n_t])

    saved_data = {
        "iteration": iteration,
        "x": expected_x,
        "m": expected_m,
        "v": expected_v,
        "x_history": x_history,
        "f_history": f_history,
        "x_avg_history": x_avg_history,
        "diff_avg_history": diff_avg_history,
        "t": test.times,
        "obs_traj": obs_traj,
    }

    with restart_file.open("wb") as handle:
        pickle.dump(saved_data, handle)


    loss = create_loss_instance(tmp_path, test)

    # Act
    f_hist, x_hist, x_avg_hist, _, _ = optimization.adam_optimizer(
        loss,
        x_copy=np.array([1, 1]),  # ignored because restart=True
        restart=True,
        restart_file=restart_file,
        max_iterations=7,  # keep short
    )

    # Assert
    np.testing.assert_allclose(f_hist[:iteration], f_history, rtol=1e-7, atol=1e-9)
    assert_list_of_arrays_equal(x_hist[:iteration], x_history)
    assert_list_of_arrays_equal(x_avg_hist[:iteration], x_avg_history)

    with pytest.raises(ValueError, match="Restart file not found"):
        _, _, _, _, _ = optimization.adam_optimizer(
            loss,
            x_copy=np.array([1, 1]),  # ignored because restart=True
            restart=True,
            restart_file=Path("/dummy/restart/file"),
            max_iterations=7,  # keep short
        )


def test_adam_selects_latest_restart_file(tmp_path: Path) -> None:

    test = Parameters()
    loss = create_loss_instance(tmp_path, test)


    def make_saved(step, t_value):
        return {
            "iteration": step,
            "x": np.array([0.0]*test.d),
            "m": np.array([0.0]*test.d),
            "v": np.array([0.0]*test.d),
            "x_history": [np.array([0.0]*test.d)],
            "f_history": [0.0],
            "x_avg_history": [np.array([0.0]*test.d)],
            "diff_avg_history": [0.0],
            "t": np.array([t_value]),
            "obs_traj": np.array([[t_value + 1.0]]),
        }

    # create files in non-sorted creation order to ensure sorting in code matters
    files_info = [
        ("restart_step_0001.pkl", make_saved(0, 1.0)),
        ("restart_step_0003.pkl", make_saved(2, 3.0)),  # this should be picked as latest
        ("restart_step_0002.pkl", make_saved(1, 2.0)),
    ]

    for name, saved in files_info:
        with (tmp_path / name).open("wb") as fh:
            pickle.dump(saved, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # Call optimizer with restart enabled and restart_file=None.
    # Set max_iterations=1 so that after loading the restart file the optimizer loop does not run.
    _, _, _, returned_t, returned_obs = optimization.adam_optimizer(loss, np.array([0.0]*test.d), restart=True, restart_file=None, max_iterations=1)

    # The optimizer should have loaded the latest restart file (restart_step_0003.pkl)
    # which contains t = [3.0] and obs_traj = [[4.0]]
    assert np.array_equal(returned_t, np.array([3.0]))
    assert np.array_equal(returned_obs, np.array([[4.0]]))



    non_pkl = tmp_path / "keep.txt"
    non_pkl.write_text("should stay")

    loss = create_loss_instance(tmp_path, test)

    # Testing all .pkl files are deleted if restart is False
    _, _, _, returned_t, returned_obs = optimization.adam_optimizer(loss, np.array([0.0]*test.d), restart=False, restart_file=None, max_iterations=1)

    # Assert all .pkl files were removed
    remaining_pkls = list(tmp_path.glob("*.pkl"))
    assert len(remaining_pkls) == 1, f"Expected 1 .pkl files, found: {remaining_pkls}"
    assert remaining_pkls[0] == tmp_path / "restart_step_0001.pkl", f"Expected {tmp_path}/restart_step_0001.pkl, found: {remaining_pkls[0]}"

    # Non-.pkl file should remain
    assert non_pkl.exists()
    assert non_pkl.read_text() == "should stay"
