from yaqs.general.libraries.noise_library import NoiseLibrary


class NoiseModel:
    # TODO: Currently assumes processes affect all sites equally
    def __init__(self, processes: list[str]=[], strengths: list[float]=[]):
        assert len(processes) == len(strengths)
        self.processes = processes
        self.strengths = strengths
        self.jump_operators = []
        for process in processes:
            self.jump_operators.append(getattr(NoiseLibrary, process)().matrix)
