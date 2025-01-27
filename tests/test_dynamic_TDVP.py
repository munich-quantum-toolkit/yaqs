import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_mps():
    """
    A minimal MPS mock with a controlled bond dimension.
    """
    from yaqs.general.data_structures.MPS import MPS

    mps = MagicMock(spec=MPS)
    return mps

@pytest.fixture
def mock_mpo():
    """
    Minimal MPO mock.
    """
    from yaqs.general.data_structures.MPO import MPO

    return MagicMock(spec=MPO)

@pytest.fixture
def mock_sim_params():
    """
    Fake simulation parameters with dt, max_bond_dim, threshold, etc.
    """
    from yaqs.general.data_structures.simulation_parameters import PhysicsSimParams

    sim_params = PhysicsSimParams(observables=[], T=1.0, dt=0.1, max_bond_dim=10, threshold=1e-6)
    return sim_params

@patch("yaqs.physics.methods.dynamic_TDVP.two_site_TDVP")
@patch("yaqs.physics.methods.dynamic_TDVP.single_site_TDVP")
def test_dynamic_tdvp_two_site(mock_single, mock_two_site, mock_mps, mock_mpo, mock_sim_params):
    """
    If current_max_bond_dim <= sim_params.max_bond_dim,
    dynamic_TDVP should call two_site_TDVP exactly once.
    """
    from yaqs.physics.methods.dynamic_TDVP import dynamic_TDVP

    # Suppose the MPS has a bond dimension of 8
    mock_mps.write_max_bond_dim.return_value = 8

    dynamic_TDVP(mock_mps, mock_mpo, mock_sim_params)

    # two_site_TDVP should be called
    mock_two_site.assert_called_once_with(mock_mps, mock_mpo, mock_sim_params.dt, 
                                          threshold=mock_sim_params.threshold, numsteps=1)
    # single_site_TDVP should not be called
    mock_single.assert_not_called()

@patch("yaqs.physics.methods.dynamic_TDVP.two_site_TDVP")
@patch("yaqs.physics.methods.dynamic_TDVP.single_site_TDVP")
def test_dynamic_tdvp_single_site(mock_single, mock_two_site, mock_mps, mock_mpo, mock_sim_params):
    """
    If current_max_bond_dim > sim_params.max_bond_dim,
    dynamic_TDVP should call single_site_TDVP exactly once.
    """
    from yaqs.physics.methods.dynamic_TDVP import dynamic_TDVP

    # Suppose the MPS bond dimension is 12, which is above the threshold of 10
    mock_mps.write_max_bond_dim.return_value = 12

    dynamic_TDVP(mock_mps, mock_mpo, mock_sim_params)

    # single_site_TDVP should be called
    mock_single.assert_called_once_with(mock_mps, mock_mpo, mock_sim_params.dt, numsteps=1)
    # two_site_TDVP should not be called
    mock_two_site.assert_not_called()
