"""Dense and MPO comb (process-tensor) wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mqt.yaqs.core.data_structures.networks import MPO

from .estimator import (
    canonicalize_upsilon,
    comb_qmi_from_upsilon_dense,
    comb_cmi_from_upsilon_dense,
    predict_from_dense_upsilon,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy.typing import NDArray


class DenseComb:
    """Wrapper around a dense comb Choi operator Υ."""

    def __init__(self, upsilon: NDArray[np.complex128], timesteps: list[float]) -> None:
        self.upsilon = upsilon
        self.timesteps = timesteps

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return the underlying dense comb matrix Υ."""
        return self.upsilon

    def canonicalize(
        self,
        *,
        hermitize: bool = True,
        psd_project: bool = False,
        normalize_trace: bool = True,
        psd_tol: float = 1e-12,
    ) -> "DenseComb":
        """Return a new DenseComb with canonicalized Υ."""
        ups_c = canonicalize_upsilon(
            self.upsilon,
            hermitize=hermitize,
            psd_project=psd_project,
            normalize_trace=normalize_trace,
            psd_tol=psd_tol,
        )
        return DenseComb(ups_c, self.timesteps)

    def predict(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Predict final state given a list of CPTP interventions."""
        return predict_from_dense_upsilon(self.upsilon, interventions)

    def qmi(self, **kwargs: object) -> float:
        """Quantum mutual information I(F:P) from this comb."""
        return comb_qmi_from_upsilon_dense(self.upsilon, **kwargs)

    def cmi(self, **kwargs: object) -> float:
        """Conditional mutual information I(F:P_{<k} | P_k) from this comb."""
        return comb_cmi_from_upsilon_dense(self.upsilon, **kwargs)


class MPOComb:
    """Wrapper around an MPO representation of a comb Choi operator Υ."""

    def __init__(self, upsilon_mpo: MPO, timesteps: list[float]) -> None:
        self.upsilon_mpo = upsilon_mpo
        self.timesteps = timesteps

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return the dense matrix representation of Υ."""
        return self.upsilon_mpo.to_matrix()

    def to_dense(self) -> DenseComb:
        """Convert the MPO comb to a DenseComb."""
        return DenseComb(self.upsilon_mpo.to_matrix(), self.timesteps)

    def qmi(self, **kwargs: object) -> float:
        """Quantum mutual information computed via dense fallback."""
        return self.to_dense().qmi(**kwargs)

    def cmi(self, **kwargs: object) -> float:
        """Conditional mutual information computed via dense fallback."""
        return self.to_dense().cmi(**kwargs)

