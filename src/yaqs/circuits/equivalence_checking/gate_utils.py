import numpy as np
import opt_einsum as oe
from qiskit.dagcircuit.dagnode import DAGOpNode

from yaqs.general.libraries.tensor_library import TensorLibrary


def apply_gate(gate: TensorLibrary, theta: np.ndarray, site0: int, site1: int, conjugate: bool = False):
    """
    Applies a single- or two-qubit gate (or multi-qubit gate) from a TensorLibrary object
    to the local tensor `theta`.

    Args:
        gate: A TensorLibrary gate object, containing a .tensor and .interaction attributes.
        theta: The local tensor to update.
        site0: The first qubit (site) index.
        site1: The second qubit (site) index.
        conjugate: Whether to apply the gate tensor in a conjugated manner.

    Returns:
        The updated local tensor after applying the gate.
    """
    import numpy as np
    import opt_einsum as oe

    # Check gate site usage
    assert gate.interaction in [1, 2], "Gate interaction must be 1 or 2."

    if gate.interaction == 1:
        assert gate.sites[0] in [site0, site1], \
            "Single-qubit gate must be on one of the sites."
    elif gate.interaction == 2:
        assert gate.sites[0] in [site0, site1] and gate.sites[1] in [site0, site1], \
            "Two-qubit gate must be on the correct pair of sites."

    # Nearest-neighbor gates (theta.ndim == 6) or long-range gates (theta.ndim == 8)
    if theta.ndim == 6:
        if conjugate:
            theta = np.transpose(theta, (3, 4, 2, 0, 1, 5))

        if gate.name == "I":
            pass  # Identity gate
        elif gate.interaction == 1:
            if gate.sites[0] == site0:
                if conjugate:
                    theta = oe.contract('ij, jklmno->iklmno', np.conj(gate.tensor), theta)
                else:
                    theta = oe.contract('ij, jklmno->iklmno', gate.tensor, theta)
            elif gate.sites[0] == site1:
                if conjugate:
                    theta = oe.contract('ij, kjlmno->kilmno', np.conj(gate.tensor), theta)
                else:
                    theta = oe.contract('ij, kjlmno->kilmno', gate.tensor, theta)
        elif gate.interaction == 2:
            if conjugate:
                theta = oe.contract('ijkl, klmnop->ijmnop', np.conj(gate.tensor), theta)
            else:
                theta = oe.contract('ijkl, klmnop->ijmnop', gate.tensor, theta)

        if conjugate:
            theta = np.transpose(theta, (3, 4, 2, 0, 1, 5))

    elif theta.ndim == 8:
        if conjugate:
            theta = np.transpose(theta, (4, 5, 3, 2, 0, 1, 6, 7))

        if gate.name == "I":
            pass  # Identity gate
        elif gate.interaction == 1:
            if gate.sites[0] == site0:
                if conjugate:
                    theta = oe.contract('ab, bcdefghi->acdefghi', np.conj(gate.tensor), theta)
                else:
                    theta = oe.contract('ab, bcdefghi->acdefghi', gate.tensor, theta)
            elif gate.sites[0] == site1:
                if conjugate:
                    theta = oe.contract('ab, cbdefghi->cadefghi', np.conj(gate.tensor), theta)
                else:
                    theta = oe.contract('ab, cbdefghi->cadefghi', gate.tensor, theta)
        elif gate.interaction == 2:
            if conjugate:
                theta = oe.contract('abcd, cdefghij->abefghij', np.conj(gate.tensor), theta)
            else:
                theta = oe.contract('abcd, cdefghij->abefghij', gate.tensor, theta)

        if conjugate:
            theta = np.transpose(theta, (4, 5, 3, 2, 0, 1, 6, 7))

    return theta