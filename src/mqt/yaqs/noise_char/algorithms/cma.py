#%%
import numpy as np
import cma



def cma_opt(f, x0, x_low=None, x_up=None, sigma0=0.01, popsize=4, max_iter=500):
    """
    CMA-ES optimization with optional lower and upper bounds per dimension
    and a maximum number of iterations.

    Parameters
    ----------
    f : callable
        Objective function to minimize.
    x0 : array-like
        Initial guess.
    sigma0 : float
        Initial standard deviation (step size).
    popsize : int, optional
        Population size.
    x_low : float or array-like, optional
        Lower bounds for each dimension (default: -inf).
    x_up : float or array-like, optional
        Upper bounds for each dimension (default: +inf).
    max_iter : int, optional
        Maximum number of iterations (default: 500).

    Returns
    -------
    fbest : float
        Best objective function value.
    xbest : ndarray
        Best solution found.
    """

    x0 = np.array(x0, dtype=float)

    # Handle flexible bounds
    if x_low is None:
        x_low = -np.inf * np.ones_like(x0)
    if x_up is None:
        x_up = np.inf * np.ones_like(x0)
    x_low = np.array(x_low, dtype=float)
    x_up = np.array(x_up, dtype=float)

    # CMA-ES configuration
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            'popsize': popsize,
            'verb_disp': 0,
            'bounds': [x_low.tolist(), x_up.tolist()],
        },
    )

    # Run optimization loop
    for i in range(max_iter):
        solutions = es.ask()
        values = [f(x) for x in solutions]
        es.tell(solutions, values)

        # Optional custom convergence detection
        if hasattr(f, "converged") and getattr(f, "converged", False):
            print(f"Average stable at iteration {i}.")
            break

        if es.stop():
            break

        if f.converged:
            print(f"Average stable at iteration {f.n_eval}.")
            break

    result = es.result
    return result.xbest, result.fbest


