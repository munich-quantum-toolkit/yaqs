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
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, BUGConfig, EvolutionMode
from mqt.yaqs.core.methods.bug import bug, bug_sweep, build_trial_basis, prepare_canonical_site_tensors
from mqt.yaqs.core.methods.decompositions import left_qr, right_qr
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
    return np.asarray(rng.standard_normal(size) + 1j * rng.standard_normal(size) / np.sqrt(2), dtype=np.complex128)


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


def _row_space_projector(rows: NDArray[np.complex128], *, tol: float = 1e-10) -> NDArray[np.complex128]:
    """Orthogonal projector onto the row space of ``rows``.

    Returns:
        The Hermitian projector onto the numerical row space.
    """
    _u, s, vh = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.sum(s > tol * max(float(s[0]), 1.0)))
    basis = vh[:rank, :]
    return basis.conj().T @ basis


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
    """BUG matches dense exact evolution on small systems."""
    mps = random_mps(shapes)
    ref = mps.to_vec().copy()
    mpo = MPO.ising(length, 1.0, 0.5)
    sim_params = AnalogSimParams(preset="exact", get_state=True, elapsed_time=1, dt=0.05)
    bug(mps, mpo, sim_params)
    exact = expm(-1j * sim_params.dt * mpo.to_matrix_mps_order()) @ ref
    np.testing.assert_allclose(mps.to_vec(), exact, rtol=1e-8, atol=1e-10)
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
    ) -> None:
        seen["threshold"] = threshold
        seen["max_bond_dim"] = max_bond_dim
        seen["trunc_mode"] = trunc_mode
        self.set_center(0)

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


def test_proposition1_full_rank_row_spaces_agree() -> None:
    """Full-rank center and explicit stacks supply the same row space before QR."""
    rng = np.random.default_rng(0)
    b0 = np.linalg.qr(rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2)))[0].T
    s_mat = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    c_pred = rng.standard_normal((2, 4)) + 1j * rng.standard_normal((2, 4))
    p_c = _row_space_projector(np.vstack([s_mat @ b0, c_pred]))
    p_e = _row_space_projector(np.vstack([b0, c_pred]))
    assert np.allclose(p_c, p_e, atol=1e-10)


def test_proposition1_rank_deficient_counterexample() -> None:
    """Center stacking can miss an old direction that explicit retention keeps."""
    b0 = np.eye(2, dtype=np.complex128)
    s_mat = np.diag([1.0, 0.0]).astype(np.complex128)
    c_pred = s_mat.copy()
    p_c = _row_space_projector(np.vstack([s_mat @ b0, c_pred]))
    p_e = _row_space_projector(np.vstack([b0, c_pred]))
    e2 = np.array([0.0, 1.0], dtype=np.complex128)
    assert np.linalg.norm(p_e @ e2) == pytest.approx(1.0, abs=1e-10)
    assert np.linalg.norm(p_c @ e2) == pytest.approx(0.0, abs=1e-10)


def test_proposition1_explicit_old_basis_retains_old_space() -> None:
    """build_trial_basis(explicit_old_basis) retains the transported old block space."""
    old_q = crandn((2, 3, 4), seed=1)
    phys, left, right = old_q.shape
    q_mat, _ = np.linalg.qr(old_q.transpose(0, 2, 1).reshape(phys * right, left))
    old_q = np.asarray(q_mat[:, :left].reshape(phys, right, left).transpose(0, 2, 1), dtype=np.complex128)
    deeper = np.eye(right, dtype=np.complex128)
    new_q, _overlap = build_trial_basis(
        old_q=old_q,
        working_center=crandn((2, 3, 4), seed=3),
        predictor=crandn((2, 3, 4), seed=2),
        deeper_overlap=deeper,
        is_endpoint=False,
        basis_mode="explicit_old_basis",
    )
    old_basis = np.tensordot(old_q, deeper, axes=(2, 0))
    old_rows = old_basis.transpose(1, 0, 2).reshape(old_basis.shape[1], -1)
    new_rows = new_q.transpose(1, 0, 2).reshape(new_q.shape[1], -1)
    projector = _row_space_projector(new_rows)
    for row in old_rows:
        assert np.linalg.norm(row - row @ projector) == pytest.approx(0.0, abs=1e-8)


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
        basis_mode="center",
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


def test_build_trial_basis_fixed_profile_no_concat() -> None:
    """fixed_profile factorizes the predictor without concatenating old tensors."""
    predictor = crandn((2, 3, 4), seed=8)
    new_q, overlap = build_trial_basis(
        old_q=crandn((2, 3, 4), seed=6),
        working_center=crandn((2, 3, 4), seed=7),
        predictor=predictor,
        deeper_overlap=np.eye(4, dtype=np.complex128),
        is_endpoint=False,
        basis_mode="fixed_profile",
    )
    ref_q, _ = left_qr(predictor)
    assert np.allclose(new_q, ref_q)
    assert overlap.shape == (3, new_q.shape[1])


def test_fixed_profile_zero_h_preserves_profile() -> None:
    """fixed_profile with compression none preserves state and bond profile for zero H."""
    mps = random_mps([(2, 1, 2), (2, 2, 3), (2, 3, 1)])
    entry = [int(mps.tensors[i].shape[2]) for i in range(mps.length - 1)]
    ref = mps.to_vec().copy()
    zero = MPO()
    zero.tensors = [np.zeros((2, 2, 1, 1), dtype=np.complex128) for _ in range(3)]
    zero.length = 3
    zero.physical_dimension = 2
    bug(
        mps,
        zero,
        AnalogSimParams(
            preset="exact",
            get_state=True,
            elapsed_time=1,
            evolution_mode=EvolutionMode.BUG,
            bug_config=BUGConfig(basis_mode="fixed_profile", compression="none"),
        ),
    )
    exit_profile = [int(mps.tensors[i].shape[2]) for i in range(mps.length - 1)]
    assert exit_profile == entry
    assert abs(np.vdot(ref, mps.to_vec())) == pytest.approx(1.0, abs=1e-10)


def test_alternating_endpoints_uses_half_dt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alternating schedule applies two positive half-steps of dt/2."""
    dts: list[float] = []

    def capture_sweep(
        state: MPS,
        _mpo: MPO,
        *,
        dt: float,
        krylov_tol: float,
        basis_mode: str = "center",
    ) -> None:
        del krylov_tol, basis_mode
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
            bug_config=BUGConfig(schedule="alternating_endpoints", compression="none"),
        ),
    )
    assert dts == pytest.approx([0.1, 0.1])
    for a, b in zip(ref_tensors, mpo.tensors, strict=True):
        assert np.allclose(a, b)


def test_compression_call_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compression placement controls how often MPS.compress is invoked."""
    counts = {"n": 0}

    def counting_compress(
        self: MPS,
        threshold: float,
        *,
        max_bond_dim: int | None = None,
        trunc_mode: str = "discarded_weight",
    ) -> None:
        del threshold, max_bond_dim, trunc_mode
        counts["n"] += 1
        self.set_center(0)

    monkeypatch.setattr(MPS, "compress", counting_compress)

    def run(config: BUGConfig) -> int:
        counts["n"] = 0
        bug(
            random_mps([(2, 1, 2), (2, 2, 1)]),
            MPO.ising(2, 1.0, 0.5),
            AnalogSimParams(
                preset="exact",
                get_state=True,
                elapsed_time=1,
                evolution_mode=EvolutionMode.BUG,
                bug_config=config,
            ),
        )
        return counts["n"]

    assert run(BUGConfig(schedule="single_endpoint", compression="after_sweep")) == 1
    assert run(BUGConfig(schedule="single_endpoint", compression="none")) == 0
    assert run(BUGConfig(schedule="alternating_endpoints", compression="after_sweep")) == 2
    assert run(BUGConfig(schedule="alternating_endpoints", compression="after_step")) == 1


def test_normalize_after_compression_opt_in() -> None:
    """normalize_after_compression=True returns unit norm."""
    mps = random_mps([(2, 1, 3), (2, 3, 1)])
    bug(
        mps,
        MPO.ising(2, 1.0, 0.5),
        AnalogSimParams(
            preset="exact",
            get_state=True,
            elapsed_time=1,
            evolution_mode=EvolutionMode.BUG,
            bug_config=BUGConfig(normalize_after_compression=True),
        ),
    )
    assert abs(mps.norm()) == pytest.approx(1.0, abs=1e-10)


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
            bug_config=BUGConfig(compression="none"),
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
        bug_config=BUGConfig(
            schedule="alternating_endpoints",
            compression="none",
        ),
        max_bond_dim=None,
        krylov_tol=1e-12,
    )
    for _ in range(round(total_time / dt)):
        bug(state, mpo, params)
    assert abs(np.vdot(reference, state.to_vec())) == pytest.approx(1.0, abs=1e-8)
    assert state.orthogonality_center == 0
    assert state.flipped is False


def test_fixed_profile_compression_may_shrink_bonds() -> None:
    """fixed_profile forbids enlargement only; post-step compression may shrink χ."""
    mps = random_mps([(2, 1, 2), (2, 2, 4), (2, 4, 2), (2, 2, 1)])
    entry = [int(mps.tensors[i].shape[2]) for i in range(mps.length - 1)]
    bug(
        mps,
        MPO.ising(4, 1.0, 0.5),
        AnalogSimParams(
            preset="exact",
            get_state=True,
            elapsed_time=1,
            dt=0.05,
            evolution_mode=EvolutionMode.BUG,
            bug_config=BUGConfig(
                basis_mode="fixed_profile",
                schedule="alternating_endpoints",
                compression="after_step",
                normalize_after_compression=True,
            ),
            trunc_mode="hard_cutoff",
            svd_threshold=0.0,
            max_bond_dim=2,
            krylov_tol=1e-12,
        ),
    )
    exit_profile = [int(mps.tensors[i].shape[2]) for i in range(mps.length - 1)]
    assert max(exit_profile) <= min(2, max(entry))
