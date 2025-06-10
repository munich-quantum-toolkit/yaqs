from monte_carlo_convergence import run_monte_carlo_convergence
from bond_dimension_convergence import run_bond_dimension_convergence
from mpdo_comparison import run_mpdo_comparison
from noise_comparison import run_noise_test
from large_scale import run_large_scale


if __name__ == "__main__":
    # 10 site Monte Carlo convergence
    # run_monte_carlo_convergence()

    # # 10 site convergence with bond dimension
    # run_bond_dimension_convergence()

    # 30 site MPDO comparison
    run_mpdo_comparison()

    # # 100 site Heisenberg model
    # run_noise_test()

    # # 1000 site Heisenberg model
    # run_large_scale()

