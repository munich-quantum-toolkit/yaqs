import pytest
import numpy as np



def test_apply_dissipation_site_canonical_0():
    """
    Check that after calling apply_dissipation, the MPS is site-canonical at site 0.
    This relies on the code's logic to shift orthogonality left after each site.
    """
    from yaqs.general.data_structures.noise_model import NoiseModel
    from yaqs.general.data_structures.MPS import MPS
    from yaqs.physics.methods.dissipation import apply_dissipation

    # 1) Create a small MPS of length 3 (for example),
    #    with random or simple product tensors.
    length = 3
    pdim = 2
    # We'll build trivial rank-3 tensors: shape => (pdim, 1, 1).
    # So it's effectively a product state with no entanglement.
    tensors = []
    for _ in range(length):
        # random 2-element vector => shape (2,1,1)
        vec = np.random.rand(pdim).astype(complex)
        vec /= np.linalg.norm(vec)  # normalize local
        tensor = vec.reshape(pdim, 1, 1)
        tensors.append(tensor)

    state = MPS(length=length, tensors=tensors, physical_dimensions=[pdim]*length)

    # 2) Create a minimal NoiseModel. Suppose we have 1 jump operator (like sigma_z or random).
    #    We set a small strength for a small dt test.
    noise_model = NoiseModel(processes=["relaxation"], strengths=[0.1])

    dt = 0.1

    # 3) Apply dissipation
    apply_dissipation(state, noise_model, dt)

    # 4) Now check that MPS is site-canonical at site 0.
    #    We'll assume state.check_canonical_form() => returns 0 or [0].
    canonical_site = state.check_canonical_form()
    # If your code returns a list with a single site [0], do:
    #   assert canonical_site == [0]
    # If it returns an int site, do:
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after apply_dissipation, "
        f"but got canonical site: {canonical_site}"
    )
