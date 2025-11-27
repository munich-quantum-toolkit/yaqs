import numpy as np

def mcmc_opt(f, x0, x_low=None, x_up=None, max_iter=500,
         step_size=0.05, step_rate=0.99, min_step_size=0, temperature=1.0, anneal_rate=0.99,
         patience=100):
    """
    MCMC-based optimization with early stopping if no improvement occurs for 'patience' iterations.

    Parameters
    ----------
    f : callable
        Objective function f(x), returns scalar.
    x0 : array-like
        Initial point.
    x_low, x_up : array-like or None
        Lower and upper bounds for x. Use None for unbounded.
    max_iter : int
        Maximum number of MCMC iterations.
    step_size : float
        Std of Gaussian proposal distribution.
    temperature : float
        Initial temperature.
    anneal_rate : float
        Cooling factor per iteration.
    patience : int
        Number of steps to allow without improvement before stopping early.

    Returns
    -------
    xbest : ndarray
        Best point found.
    fbest : float
        Best function value.
    """

    x = np.array(x0, dtype=float)
    ndim = x.size

    fx = f(x)
    xbest, fbest = x.copy(), fx

    if x_low is not None:
        x_low = np.array(x_low, dtype=float)
    if x_up is not None:
        x_up = np.array(x_up, dtype=float)

    no_improve_counter = 0

    for i in range(max_iter):

        # Gaussian proposal
        x_new = x + np.random.normal(scale=step_size, size=ndim)

        # Apply bounds
        if x_low is not None:
            x_new = np.maximum(x_new, x_low)
        if x_up is not None:
            x_new = np.minimum(x_new, x_up)

        f_new = f(x_new)

        # Metropolis–Hastings acceptance
        delta = f_new - fx
        acceptance_prob = np.exp(-delta / temperature)

        if np.random.rand() < acceptance_prob:
            x, fx = x_new, f_new

        # Track global best
        if fx < fbest:
            xbest, fbest = x.copy(), fx
            no_improve_counter = 0  # reset
        else:
            no_improve_counter += 1

        # Early stopping condition
        if no_improve_counter >= patience:
            break

        # Annealing
        temperature *= anneal_rate

        step_size *= step_rate

        step_size = max(step_size, 1e-3)

        # print(f"Iter {i+1}/{max_iter}, x_best: {xbest}, Best f: {fbest:.6f}, Temp: {temperature:.4f}, Step size: {step_size:.4f}")

    return xbest, fbest
