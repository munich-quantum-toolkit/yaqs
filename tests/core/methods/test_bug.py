# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Basis-Update Galerkin (BUG) method."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.linalg import expm

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, EvolutionMode
from mqt.yaqs.core.methods.bug import bug, bug_sweep, build_trial_basis, prepare_canonical_site_tensors
from mqt.yaqs.core.methods.decompositions import right_qr
from mqt.yaqs.core.methods.tdvp.primitives import update_left_environment

if TYPE_CHECKING:
    from numpy.typing import NDArray


def crandn(
    size: int | tuple[int, ...], *args: int, seed: np.random.Generator | int | None = None
) -> NDArray[np.complex128]:
    """Draw random samples from the standard complex normal distribution.

    Returns:
        Complex array with the requested shape.
    """
    if isinstance(size, int) and len(args) > 0:
        size = (size, *list(args))
    elif isinstance(size, int):
        size = (size,)
    rng = np.random.default_rng(seed)
    return np.asarray((rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2), dtype=np.complex128)


def random_mps(shapes: list[tuple[int, int, int]]) -> MPS:
    """Create a normalized random MPS with the given shapes.

    Returns:
        A normalized :class:`MPS`.
    """
    mps = MPS(len(shapes), tensors=[crandn(shape) for shape in shapes])
    mps.normalize()
    return mps


def _is_right_isometric(tensor: NDArray[np.complex128], *, atol: float = 1e-10) -> bool:
    """Return True if contracting phys+right yields identity on the left bond.

    Returns:
        Whether the tensor is right-isometric within ``atol``.
    """
    left = tensor.shape[1]
    gram = np.tensordot(tensor, tensor.conj(), axes=([0, 2], [0, 2]))
    return bool(np.allclose(gram, np.eye(left), atol=atol))


def test_prepare_canonical_site_tensors_single_site() -> None:
    """Preparation for a length-1 MPS leaves the physical tensor unchanged."""
    mps_tensor = crandn(2, 3, 4)
    mps = MPS(1, tensors=[mps_tensor])
    ref_mps = deepcopy(mps)
    mpo = MPO()
    mpo.custom([crandn(2, 2, 1, 1)])
    canon_sites, left_envs = prepare_canonical_site_tensors(mps, mpo)
    assert mps.almost_equal(ref_mps)
    assert np.allclose(left_envs[0], np.eye(3).reshape(3, 1, 3))
    assert np.allclose(canon_sites[0], mps_tensor)


def test_prepare_canonical_site_tensors_three_sites() -> None:
    """Preparation for a length-3 MPS matches the explicit QR reference."""
    mps_tensors = [crandn(shape) for shape in [(2, 3, 4), (2, 4, 5), (2, 5, 3)]]
    mps = MPS(3, tensors=mps_tensors)
    ref_mps = deepcopy(mps)
    mpo_tensors = [crandn(shape) for shape in [(2, 2, 1, 3), (2, 2, 3, 4), (2, 2, 4, 1)]]
    mpo = MPO()
    mpo.custom(mpo_tensors, transpose=False)
    canon_sites, left_envs = prepare_canonical_site_tensors(mps, mpo)
    assert mps.almost_equal(ref_mps)

    assert np.allclose(left_envs[0], np.eye(3, dtype=np.complex128).reshape(3, 1, 3))
    assert np.allclose(canon_sites[0], mps_tensors[0])

    q_last, r_matrix = right_qr(mps_tensors[0])
    correct_canon = np.tensordot(r_matrix, mps_tensors[1], axes=(1, 1)).transpose(1, 0, 2)
    correct_env = update_left_environment(q_last, q_last, mpo_tensors[0], left_envs[0])
    assert np.allclose(correct_env, left_envs[1])
    assert np.allclose(correct_canon, canon_sites[1])

    q_last, r_matrix = right_qr(np.asarray(correct_canon, dtype=np.complex128))
    correct_canon = np.tensordot(r_matrix, mps_tensors[2], axes=(1, 1)).transpose(1, 0, 2)
    correct_env = update_left_environment(q_last, q_last, mpo_tensors[1], left_envs[1])
    assert np.allclose(correct_env, left_envs[2])
    assert np.allclose(correct_canon, canon_sites[2])


def test_prepare_does_not_alias_physical_tensors() -> None:
    """Preparation must not mutate non-root physical tensors."""
    mps = random_mps([(2, 1, 3), (2, 3, 2), (2, 2, 1)])
    originals = [t.copy() for t in mps.tensors]
    ids = [id(t) for t in mps.tensors]
    canon_sites, _left = prepare_canonical_site_tensors(mps, MPO.ising(3, 1.0, 0.5))
    for orig, live in zip(originals, mps.tensors, strict=True):
        assert np.allclose(orig, live)
    assert id(canon_sites[0]) == ids[0]
    assert id(canon_sites[1]) != ids[1]


@pytest.mark.parametrize(
    ("shapes", "length"),
    [
        ([(2, 1, 1)], 1),
        ([(2, 1, 3), (2, 3, 1)], 2),
        ([(2, 1, 4), (2, 4, 4), (2, 4, 1)], 3),
    ],
)
def test_bug_dense_reference(shapes: list[tuple[int, int, int]], length: int) -> None:
    """BUG matches dense exact evolution on small systems (up to global phase)."""
    mps = random_mps(shapes)
    ref = mps.to_vec().copy()
    mpo = MPO.ising(length, 1.0, 0.5)
    sim_params = AnalogSimParams(preset="exact", get_state=True, elapsed_time=1, dt=0.05)
    bug(mps, mpo, sim_params)
    exact = expm(-1j * sim_params.dt * mpo.to_matrix_mps_order()) @ ref
    assert abs(np.vdot(exact, mps.to_vec())) == pytest.approx(1.0, abs=1e-8)
    assert mps.orthogonality_center == 0


def test_bug_requires_center_at_zero() -> None:
    """BUG rejects an MPS whose tracked center is not at site 0."""
    mps = random_mps([(2, 1, 4), (2, 4, 4), (2, 4, 1)])
    mps.set_center(1)
    with pytest.raises(ValueError, match="bug"):
        bug(mps, MPO.ising(3, 1, 0.5), AnalogSimParams(preset="exact", get_state=True, elapsed_time=1))


def test_bug_sweep_rejects_unknown_center() -> None:
    """The uncompressed kernel rejects a missing orthogonality center."""
    mps = random_mps([(2, 1, 4), (2, 4, 1)])
    mps.set_center(None)
    with pytest.raises(ValueError, match="bug_sweep"):
        bug_sweep(mps, MPO.ising(2, 1, 0.5), dt=0.1, krylov_tol=1e-10)


def test_bug_length_mismatch_raises() -> None:
    """BUG rejects Hamiltonians whose site count differs from the MPS."""
    with pytest.raises(ValueError, match="same number of sites"):
        bug(MPS(2, state="zeros"), MPO.ising(3, 1, 0.5), AnalogSimParams(preset="exact", elapsed_time=1))


def test_bug_sweep_no_compression_and_isometric_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """bug_sweep never compresses and leaves non-root sites right-isometric."""

    def boom(*_args: object, **_kwargs: object) -> None:
        msg = "compress should not be called from bug_sweep"
        raise AssertionError(msg)

    monkeypatch.setattr(MPS, "compress", boom)
    mps = random_mps([(2, 1, 3), (2, 3, 4), (2, 4, 1)])
    bug_sweep(mps, MPO.ising(3, 1.0, 0.5), dt=0.05, krylov_tol=1e-12)
    assert mps.orthogonality_center == 0
    assert _is_right_isometric(mps.tensors[1])
    assert _is_right_isometric(mps.tensors[2])


def test_bug_sweep_zero_hamiltonian_preserves_state() -> None:
    """A zero Hamiltonian leaves the physical state unchanged up to gauge."""
    mps = random_mps([(2, 1, 3), (2, 3, 1)])
    ref = mps.to_vec().copy()
    zero = MPO()
    zero.tensors = [np.zeros((2, 2, 1, 1), dtype=np.complex128) for _ in range(2)]
    zero.length = 2
    zero.physical_dimension = 2
    bug_sweep(mps, zero, dt=0.2, krylov_tol=1e-12)
    assert abs(np.vdot(ref, mps.to_vec())) == pytest.approx(1.0, abs=1e-10)


def test_bug_forwards_trunc_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """bug() forwards sim_params.trunc_mode into MPS.compress."""
    seen: dict[str, object] = {}

    def fake_compress(
        self: MPS,
        threshold: float,
        *,
        max_bond_dim: int | None = None,
        trunc_mode: str = "discarded_weight",
        min_keep: int = 1,
        canonicalize: bool = True,
        restore_center: bool = True,
    ) -> None:
        seen["threshold"] = threshold
        seen["max_bond_dim"] = max_bond_dim
        seen["trunc_mode"] = trunc_mode
        seen["min_keep"] = min_keep
        seen["canonicalize"] = canonicalize
        seen["restore_center"] = restore_center
        self.set_center(self.length - 1)

    monkeypatch.setattr(MPS, "compress", fake_compress)
    bug(
        random_mps([(2, 1, 2), (2, 2, 1)]),
        MPO.ising(2, 1.0, 0.5),
        AnalogSimParams(
            preset="exact",
            get_state=True,
            elapsed_time=1,
            trunc_mode="relative",
            svd_threshold=1e-8,
            max_bond_dim=4,
        ),
    )
    assert seen["trunc_mode"] == "relative"
    assert seen["threshold"] == pytest.approx(1e-8)
    assert seen["max_bond_dim"] == 4
    assert seen["min_keep"] == 2
    assert seen["canonicalize"] is False
    assert seen["restore_center"] is False


def test_proposition2_overlap_frobenius_identity() -> None:
    """Overlap transport satisfies the local Frobenius identity."""
    rng = np.random.default_rng(4)
    b0 = np.linalg.qr(rng.standard_normal((5, 3)) + 1j * rng.standard_normal((5, 3)))[0].T
    b1 = np.linalg.qr(rng.standard_normal((5, 4)) + 1j * rng.standard_normal((5, 4)))[0].T
    overlap = b0 @ b1.conj().T
    x = rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))
    defect = np.eye(b0.shape[1]) - b1.conj().T @ b1
    left = np.linalg.norm(x @ overlap, "fro") ** 2
    right = np.linalg.norm(x, "fro") ** 2 - np.linalg.norm(x @ b0 @ defect, "fro") ** 2
    assert left == pytest.approx(right, abs=1e-10)


def test_proposition2_build_trial_basis_overlap_is_block_contraction() -> None:
    """Implemented overlap factor equals the old/new right-block contraction."""
    old_q = crandn((2, 3, 4), seed=11)
    phys, left, right = old_q.shape
    q_mat, _ = np.linalg.qr(old_q.transpose(0, 2, 1).reshape(phys * right, left))
    old_q = np.asarray(q_mat[:, :left].reshape(phys, right, left).transpose(0, 2, 1), dtype=np.complex128)
    deeper = np.linalg.qr(crandn((right, right), seed=12))[0]
    # Concatenation is on the left bond; right bond must match the retained tensor.
    predictor = crandn((2, 3, 4), seed=13)
    working = crandn((2, 3, 4), seed=14)
    new_q, overlap = build_trial_basis(
        old_q=old_q,
        working_center=working,
        predictor=predictor,
        deeper_overlap=deeper,
        is_endpoint=False,
    )
    old_basis_current = np.tensordot(old_q, deeper, axes=(2, 0))
    direct = np.tensordot(old_basis_current, new_q.conj(), axes=([0, 2], [0, 2]))
    assert np.allclose(overlap, direct, atol=1e-12)


def test_proposition2_transported_center_matches_projection() -> None:
    """Coefficient transport X↦XM matches projection into the new block space."""
    rng = np.random.default_rng(15)
    b0 = np.linalg.qr(rng.standard_normal((6, 3)) + 1j * rng.standard_normal((6, 3)))[0].T
    extra = rng.standard_normal((2, 6)) + 1j * rng.standard_normal((2, 6))
    b1 = np.linalg.qr(np.vstack([b0, extra]).T)[0].T
    overlap = b0 @ b1.conj().T
    x = rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))
    transported = (x @ overlap) @ b1
    projected = x @ b0 @ (b1.conj().T @ b1)
    assert np.allclose(transported, projected, atol=1e-10)
    # Inclusion ⇒ norm preservation for this X.
    assert np.linalg.norm(x @ overlap, "fro") == pytest.approx(np.linalg.norm(x, "fro"), abs=1e-10)


def test_alternating_endpoints_uses_half_dt(monkeypatch: pytest.MonkeyPatch) -> None:
    """BUG applies two positive half-steps of dt/2."""
    dts: list[float] = []

    def capture_sweep(
        state: MPS,
        _mpo: MPO,
        *,
        dt: float,
        krylov_tol: float,
    ) -> None:
        del krylov_tol
        dts.append(dt)
        state.set_center(0)

    monkeypatch.setattr("mqt.yaqs.core.methods.bug.bug_sweep", capture_sweep)
    mpo = MPO.ising(2, 1.0, 0.5)
    ref_tensors = [t.copy() for t in mpo.tensors]
    bug(
        random_mps([(2, 1, 2), (2, 2, 1)]),
        mpo,
        AnalogSimParams(
            preset="exact",
            get_state=True,
            elapsed_time=1,
            dt=0.2,
            evolution_mode=EvolutionMode.BUG,
        ),
    )
    assert dts == pytest.approx([0.1, 0.1])
    for a, b in zip(ref_tensors, mpo.tensors, strict=True):
        assert np.allclose(a, b)


def test_compression_after_each_half_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """BUG compresses after both halves of the alternating composition."""
    counts = {"n": 0}

    def counting_compress(
        self: MPS,
        threshold: float,
        *,
        max_bond_dim: int | None = None,
        trunc_mode: str = "discarded_weight",
        min_keep: int = 1,
        canonicalize: bool = True,
        restore_center: bool = True,
    ) -> None:
        del threshold, max_bond_dim, trunc_mode
        assert min_keep == 2
        assert canonicalize is False
        assert restore_center is False
        counts["n"] += 1
        self.set_center(self.length - 1)

    monkeypatch.setattr(MPS, "compress", counting_compress)
    bug(
        random_mps([(2, 1, 2), (2, 2, 1)]),
        MPO.ising(2, 1.0, 0.5),
        AnalogSimParams(
            preset="exact",
            get_state=True,
            elapsed_time=1,
            evolution_mode=EvolutionMode.BUG,
        ),
    )
    assert counts["n"] == 2


def test_bug_checkpoint_order_and_orientation() -> None:
    """BUG exposes the two sweeps and two compressions in execution order."""
    observed: list[tuple[str, bool, int | None]] = []

    def checkpoint(name: str, state: MPS, *, reflected: bool) -> None:
        observed.append((name, reflected, state.orthogonality_center))

    mps = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    bug(
        mps,
        MPO.ising(3, 1.0, 0.5),
        AnalogSimParams(preset="exact", elapsed_time=0.1, dt=0.1, max_bond_dim=4),
        checkpoint=checkpoint,
    )
    assert observed == [
        ("first_half_sweep", False, 0),
        ("first_compression", False, 2),
        ("second_half_sweep", True, 0),
        ("second_compression", True, 2),
    ]
    assert mps.orthogonality_center == 0
    assert mps.flipped is False


def test_normalize_after_compression() -> None:
    """BUG returns a unit-norm state after compression."""
    mps = random_mps([(2, 1, 3), (2, 3, 1)])
    bug(
        mps,
        MPO.ising(2, 1.0, 0.5),
        AnalogSimParams(
            preset="exact",
            get_state=True,
            elapsed_time=1,
            evolution_mode=EvolutionMode.BUG,
        ),
    )
    assert abs(mps.norm()) == pytest.approx(1.0, abs=1e-10)


def test_bug_normalizes_without_full_chain_normalize(monkeypatch: pytest.MonkeyPatch) -> None:
    """BUG rescales the canonical center instead of normalizing the full MPS."""
    mps = random_mps([(2, 1, 3), (2, 3, 1)])

    def reject_full_normalize(*_args: object, **_kwargs: object) -> None:
        msg = "BUG should normalize only its canonical center tensor"
        raise AssertionError(msg)

    monkeypatch.setattr(MPS, "normalize", reject_full_normalize)
    bug(
        mps,
        MPO.ising(2, 1.0, 0.5),
        AnalogSimParams(preset="exact", get_state=True, elapsed_time=0.1, dt=0.1),
    )
    assert abs(mps.norm()) == pytest.approx(1.0, abs=1e-10)


def test_bug_reuses_compression_endpoint_after_reflection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normal BUG execution needs no gauge sweep between compression and reflection."""

    def reject_center_recovery(*_args: object, **_kwargs: object) -> None:
        msg = "endpoint-aware BUG should not need center recovery"
        raise AssertionError(msg)

    monkeypatch.setattr("mqt.yaqs.core.methods.bug._move_center_to_zero", reject_center_recovery)
    mps = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    bug(
        mps,
        MPO.ising(3, 1.0, 0.5),
        AnalogSimParams(preset="exact", get_state=True, elapsed_time=0.1, dt=0.1),
    )
    assert mps.orthogonality_center == 0


def test_bug_asymmetric_matches_mps_ordered_dense_reference() -> None:
    """BUG agrees with dense evolution for an asymmetric Hamiltonian in MPS order."""
    length = 3
    mpo = MPO()
    mpo.from_pauli_sum(
        terms=[(1.0, "Z0"), (0.3, "X1"), (0.7, "Y2")],
        length=length,
        tol=0.0,
        n_sweeps=0,
    )
    mps = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    ref = mps.to_vec().copy()
    dt = 0.05
    bug(
        mps,
        mpo,
        AnalogSimParams(
            preset="exact",
            get_state=True,
            elapsed_time=1,
            dt=dt,
            evolution_mode=EvolutionMode.BUG,
            max_bond_dim=None,
            krylov_tol=1e-12,
        ),
    )
    exact = expm(-1j * dt * mpo.to_matrix_mps_order()) @ ref
    exact_msb = expm(-1j * dt * mpo.to_matrix()) @ ref
    assert abs(np.vdot(exact, mps.to_vec())) == pytest.approx(1.0, abs=1e-8)
    # Historical MSB dense layout must not be used as an MPS reference here.
    assert abs(np.vdot(exact_msb, mps.to_vec())) < 1.0 - 1e-3


def test_alternating_asymmetric_matches_dense_without_mock() -> None:
    """Alternating endpoints evolves an asymmetric H without mocking sweeps."""
    length = 3
    mpo = MPO()
    mpo.from_pauli_sum(
        terms=[(1.1, "Z0"), (0.4, "X1"), (0.8, "Y2")],
        length=length,
        tol=0.0,
        n_sweeps=0,
    )
    mps = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    total_time = 0.2
    dt = 0.05
    reference = expm(-1j * total_time * mpo.to_matrix_mps_order()) @ mps.to_vec()
    state = deepcopy(mps)
    params = AnalogSimParams(
        preset="exact",
        get_state=True,
        elapsed_time=total_time,
        dt=dt,
        evolution_mode=EvolutionMode.BUG,
        max_bond_dim=None,
        krylov_tol=1e-12,
    )
    for _ in range(round(total_time / dt)):
        bug(state, mpo, params)
    assert abs(np.vdot(reference, state.to_vec())) == pytest.approx(1.0, abs=1e-8)
    assert state.orthogonality_center == 0
    assert state.flipped is False
