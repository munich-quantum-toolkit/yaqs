import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    LogExpectedImprovement
)
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize, Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

import numpy as np

# --------------------------------------------
# Select acquisition function
# --------------------------------------------
def get_acquisition_function(name, model, best_f=None, beta=2.0):
    name = name.upper()
    if name == "EI":
        return ExpectedImprovement(model=model, best_f=best_f, maximize=True)
    elif name == "LEI":
        return LogExpectedImprovement(model=model, best_f=best_f, maximize=True)
    elif name == "PI":
        return ProbabilityOfImprovement(model=model, best_f=best_f, maximize=True)
    elif name == "UCB":
        return UpperConfidenceBound(model=model, beta=beta)
    else:
        raise ValueError(f"Unknown acquisition function: {name}")






# --------------------------------------------
# Bayesian Optimization Loop
# --------------------------------------------
def bayesian_opt(
    f,
    x_low,
    x_up,
    n_init=5,
    n_iter=15,
    acq_name="EI",
    std=1e-6,
    beta=2.0,
    dtype=torch.double,
    device="cpu",
):
    """
    Bayesian Optimization for MINIMIZATION with fixed noise level.

    Args:
        f: Callable[[np.ndarray], float or np.ndarray]
            Function to minimize. Must accept NumPy arrays.
        bounds: torch.tensor([[x1_min, ..., xd_min], [x1_max, ..., xd_max]])
        n_init: Number of initial random evaluations
        n_iter: Number of BO iterations
        acq_name: "EI", "PI", or "UCB"
        std: Known standard deviation of noise
    """


    bounds = torch.tensor(np.array([x_low, x_up]), dtype=torch.double)


    d = bounds.shape[1]

    # Normalized [0,1]^d → real-space bounds
    def scale_to_bounds(X_unit):
        return bounds[0] + (bounds[1] - bounds[0]) * X_unit

    # -----------------------
    # Helper: evaluate f safely
    # -----------------------
    def eval_function(X):
        """
        X: torch.Tensor of shape (n, d)
        returns: torch.Tensor of shape (n, 1)
        """
        X_np = X.detach().cpu().numpy()
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        y_np = np.array([f(xi) for xi in X_np], dtype=np.float64)
        return torch.tensor(y_np, dtype=dtype, device=device).unsqueeze(-1)

    # -----------------------
    # Initial data
    # -----------------------
    X_train = torch.rand(n_init, d, dtype=dtype, device=device)
    y = eval_function(scale_to_bounds(X_train))
    Y_train = -y  # Negate for minimization (BO maximizes internally)

    # Constant noise variance
    Yvar_train = torch.full_like(Y_train, std**2)

    # -----------------------
    # BO loop
    # -----------------------
    for i in range(n_iter):
        model = SingleTaskGP(
            X_train,
            Y_train,
            Yvar_train,
            input_transform=Normalize(d),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        best_f = Y_train.max()
        acq_func = get_acquisition_function(acq_name, model, best_f=best_f, beta=beta)

        new_x_unit, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack(
                [torch.zeros(d, device=device), torch.ones(d, device=device)]
            ),
            q=1,
            num_restarts=5,
            raw_samples=50,
        )

        new_y = eval_function(scale_to_bounds(new_x_unit))
        new_y = -new_y  # negate for minimization

        # Append new data
        X_train = torch.cat([X_train, new_x_unit])
        Y_train = torch.cat([Y_train, new_y])
        Yvar_train = torch.cat(
            [Yvar_train, torch.full_like(new_y, std**2)]
        )

        best_idx = torch.argmax(Y_train)
        best_x = scale_to_bounds(X_train[best_idx])
        best_y = -Y_train[best_idx]

        print(f"Iter {i+1:02d} | Best Loss (min): {-Y_train.max().item():.6f} |  Best x: {best_x}")


        if f.converged:
            print(f"Average stable at iteration {f.n_eval}.")
            break

    # -----------------------
    # Return best found point
    # -----------------------
      # flip back to minimization scale

    return best_x.numpy(), best_y.numpy()[0], X_train, -Y_train  # flip Y back to original scale



