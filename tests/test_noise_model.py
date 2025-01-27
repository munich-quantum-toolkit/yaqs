import pytest
from yaqs.general.data_structures.noise_model import NoiseModel


def test_noise_model_creation():
    processes = ["relaxation", "dephasing"]
    strengths = [0.1, 0.05]

    model = NoiseModel(processes, strengths)

    assert model.processes == processes
    assert model.strengths == strengths
    assert len(model.jump_operators) == len(processes)

    assert model.jump_operators[0].shape == (2,2), "First jump operator should be 2x2 for 'dephasing'."


def test_noise_model_assertion():
    """If processes and strengths differ in length, an AssertionError should be raised."""
    processes = ["relaxation", "dephasing"]
    strengths = [0.1]
    with pytest.raises(AssertionError):
        _ = NoiseModel(processes, strengths)


def test_noise_model_empty():
    """Check that NoiseModel handles an empty process list without error."""
    model = NoiseModel([], [])
    assert model.processes == []
    assert model.strengths == []
    assert model.jump_operators == []
