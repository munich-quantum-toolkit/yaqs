import numpy as np

from typing import TYPE_CHECKING
from pathlib import Path
import pickle  # noqa: S403
import contextlib
import time


from mqt.yaqs.noise_char.loss import LossClass

if TYPE_CHECKING:
    from numpy.typing import NDArray


def gradient_descent_opt(
    f: LossClass,
    x_copy: np.ndarray,
    *,
    x_low: np.ndarray | None = None,
    x_up: np.ndarray | None = None,
    alpha: float = 0.05,
    max_iter: int = 1000,
    threshold: float = 5e-4,
    max_n_convergence: int = 50,
    tolerance: float = 1e-8,
    restart: bool = False,
    restart_file: Path | None = None,
) -> tuple[
    list[float],  # f.f_history: History of loss values.
    list[np.ndarray],  # f.x_history: History of parameter vectors.
    list[np.ndarray],  # f.x_avg_history: History of averaged parameter vectors.
    np.ndarray,  # f.t: Time array from the loss function.
    np.ndarray,  # f.exp_vals_traj: Optimized trajectory of expectation values.
]:
    """Performs Adam optimization on a given loss function with support for checkpointing and restart.

    Args:
        f (loss_class): An instance of a loss function class.
        x_copy (np.ndarray): Initial parameter vector to optimize.
        alpha (float, optional): Learning rate for Adam optimizer. Default is 0.05.
        max_iter (int, optional): Maximum number of optimization iterations. Default is 1000.
        threshold (float, optional): Threshold for parameter convergence check. Default is 5e-4.
        max_n_convergence (int, optional): Number of consecutive iterations to check for convergence. Default is 50.
        tolerance (float, optional): Absolute loss tolerance for early stopping. Default is 1e-8.
        restart (bool, optional): Whether to restart optimization from a checkpoint. Default is False.
        restart_file (str, optional): Path to a specific checkpoint file to restart from.
        If None, the latest checkpoint in the working directory is used.

    Returns:
        Tuple[
            List[float],      # f.f_history: History of loss values.
            List[np.ndarray], # f.x_history: History of parameter vectors.
            List[np.ndarray], # f.x_avg_history: History of averaged parameter vectors.
            np.ndarray,       # f.t: Time array from the loss function.
            np.ndarray        # f.exp_vals_traj: Optimized trajectory of expectation values.
        ]

    Raises:
        ValueError: If restart is True but no valid checkpoint file is found.

    Notes:
        - The optimizer saves a checkpoint at every iteration in the working directory specified by `f.work_dir`.
        - All parameter values are clipped to the [0, 1] range after each update.
        - Performance metrics are logged to 'performance_metric_sec.txt' in the working directory.
    """
    restart_dir = f.work_dir

    perf_path = Path(f.work_dir) / "performance_metric_sec.txt"

    # Find the latest restart file in the restart_dir
    if restart_file is None and restart and Path(restart_dir).is_dir():
        restart_files = [
            file_path.name
            for file_path in Path(restart_dir).iterdir()
            if file_path.name.startswith("restart_step_") and file_path.suffix == ".pkl"
        ]
        if restart_files:
            # Sort by step number
            restart_files.sort()
            restart_file = Path(restart_dir) / restart_files[-1]

    # Initialization
    if restart and restart_file is not None:
        if not restart_file.exists():
            msg = "Restart file not found."
            raise ValueError(msg)
        # We only load restart files we created ourselves; no untrusted input here.
        with restart_file.open("rb") as handle:
            saved = pickle.load(handle)  # noqa: S301
        x = saved["x"]
        start_iter = saved["iteration"] + 1  # resume from next iteration

        f.set_history(saved["x_history"], saved["f_history"], saved["x_avg_history"], saved["diff_avg_history"])

        f.t = saved["t"].copy()
        f.obs_array = saved["obs_traj"].copy()

    else:
        # Remove all .pkl files in the folder
        restart_path = Path(restart_dir)
        for file_path in restart_path.iterdir():
            if file_path.suffix == ".pkl":
                with contextlib.suppress(Exception):
                    file_path.unlink()

        x = x_copy.copy()
        d = len(x)
        start_iter = 0

        # Write a header to performance_metric.txt in f.work_dir

        with perf_path.open("w", encoding="utf-8") as pf:
            pf.write("# iter    opt_step_time    simulation_time    avg_traj_time    min_traj_time    max_traj_time\n")

    for i in range(start_iter, max_iter):
        # Calculate loss and gradients (unchanged)

        start_time = time.time()

        loss, grad, sim_time = f(x)

        update = alpha*grad

        print("Updates: ",x, alpha, grad, update)

        # Update simulation parameters with Adam update (NEW)
        x -= update
        
        # Ensure x stays in bounds (NEW)
        if x_low is not None and x_up is not None:
            x = np.clip(x, x_low, x_up)

        print("Updated x",x)

        restart_data = {
            "iteration": i,
            "x": x.copy(),
            "x_loss": x + update,
            "loss": loss,
            "grad": grad.copy(),
            "update": update.copy(),
            "f_history": f.f_history.copy(),
            "x_history": f.x_history.copy(),
            "x_avg_history": f.x_avg_history.copy(),
            "diff_avg_history": f.diff_avg_history.copy(),
            "t": f.t.copy(),
            "obs_traj": f.obs_array.copy(),
        }

        restart_path = Path(restart_dir) / f"restart_step_{i + 1:04d}.pkl"
        with restart_path.open("wb") as handle:
            pickle.dump(restart_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        end_time = time.time()

        iter_time = end_time - start_time

        with perf_path.open("a", encoding="utf-8") as pf:
            pf.write(f"  {i}    {iter_time}    {sim_time}  \n")

        if abs(loss) < tolerance:
            break

        # Convergence check
        if len(f.diff_avg_history) > max_n_convergence and all(
            diff < threshold for diff in f.diff_avg_history[-max_n_convergence:]
        ):
            break

    return f.f_history, f.x_history, f.x_avg_history, f.t, f.obs_array 
