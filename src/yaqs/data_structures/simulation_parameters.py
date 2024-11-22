class SimulationParams:
    def __init__(self, measurements: dict[str, int], T: float, dt: float, N: int, max_bond_dim: int, threshold: float):
        self.measurements = measurements
        self.T = T
        self.dt = dt
        self.N = N
        self.max_bond_dim = max_bond_dim
        self.threshold = threshold
