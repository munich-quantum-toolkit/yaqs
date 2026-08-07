# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Experiment-local tracing of retained MPS resources.

The paper compares retained MPS storage, not transient BLAS workspaces.  This
module therefore records the full-chain MPS after every state-changing SVD and,
for MPO application, immediately after the full MPO--MPS contraction.  The
instrumentation is deliberately local to the experiment: production YAQS
callables are monkeypatched only while :class:`ResourceTracer` is active and
are restored even when an update raises.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from types import TracebackType

    from mqt.yaqs.core.data_structures.mps import MPS


_RESERVED_ROW_FIELDS = {
    "checkpoint",
    "checkpoint_index",
    "checkpoint_in_gate",
    "n_sites",
    "parameter_count",
    "peak_bond_dim",
    "bond_dimensions",
    "updated_sites",
    "local_gate_name",
}
_ACTIVE_TRACER: ResourceTracer | None = None


def _tensor_resources(tensors: Sequence[Any]) -> tuple[int, int, list[int]]:
    """Return ``(P, chi_peak, bond_profile)`` for compatible MPS tensors.

    Args:
        tensors: Full-chain MPS tensors in ``(physical, left, right)`` order.

    Returns:
        Retained complex parameter count, maximum virtual-bond dimension, and
        the ``N+1``-entry virtual-bond profile.

    Raises:
        ValueError: If the tensor sequence is empty, contains a non-rank-three
            tensor, or has inconsistent neighboring virtual dimensions.
    """
    if not tensors:
        msg = "Cannot measure an empty MPS."
        raise ValueError(msg)

    shapes: list[tuple[int, int, int]] = []
    for site, tensor in enumerate(tensors):
        shape = tuple(int(dim) for dim in tensor.shape)
        if len(shape) != 3:
            msg = f"MPS tensor at site {site} has rank {len(shape)} instead of 3."
            raise ValueError(msg)
        shapes.append(shape)

    for site in range(len(shapes) - 1):
        if shapes[site][2] != shapes[site + 1][1]:
            msg = (
                f"Inconsistent MPS bond {site}: right dimension {shapes[site][2]} "
                f"!= left dimension {shapes[site + 1][1]}."
            )
            raise ValueError(msg)

    parameter_count = sum(physical * left * right for physical, left, right in shapes)
    bond_dimensions = [shapes[0][1], *(shape[2] for shape in shapes)]
    return parameter_count, max(bond_dimensions), bond_dimensions


def retained_mps_resources(state: MPS) -> tuple[int, int, list[int]]:
    """Measure retained full-chain MPS storage.

    Args:
        state: MPS to inspect without modifying it.

    Returns:
        ``(P, chi_peak, bond_profile)``, where ``P`` is the sum of all tensor
        sizes and ``bond_profile`` includes both boundary bonds.
    """
    return _tensor_resources(state.tensors)


@dataclass
class _GateScope:
    """Metadata and local event counter for one physical circuit gate."""

    fields: dict[str, Any]
    checkpoint_count: int = 0


@dataclass(frozen=True)
class _WindowBinding:
    """Relation between a TDVP window MPS and its full-chain parent."""

    parent: Any
    first_site: int
    last_site: int


@dataclass
class _CompressionFrame:
    """Current prospective SVD position in one MPS compression sweep."""

    state: Any
    next_site: int = 0


@dataclass(frozen=True)
class _Patch:
    """One monkeypatch to restore on context exit."""

    owner: Any
    attribute: str
    original: Any


@dataclass
class ResourceTracer:
    """Trace full-chain retained MPS resources during circuit gate updates.

    Use one tracer per process and wrap each physical circuit gate in
    :meth:`gate_scope`.  Rows use zero-based site indices, matching YAQS; the
    analysis layer can convert them to the paper's one-based convention.

    Attributes:
        rows: One flat, JSON-compatible dictionary per resource checkpoint.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    _patches: list[_Patch] = field(default_factory=list, init=False, repr=False)
    _installed: bool = field(default=False, init=False, repr=False)
    _scope: ContextVar[_GateScope | None] = field(init=False, repr=False)
    _active_mpo_state: ContextVar[Any | None] = field(init=False, repr=False)
    _compression: ContextVar[_CompressionFrame | None] = field(init=False, repr=False)
    _tdvp_windows: weakref.WeakKeyDictionary[Any, _WindowBinding] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Create context-local tracing state."""
        suffix = f"{id(self):x}"
        self._scope = ContextVar(f"circuit_resource_scope_{suffix}", default=None)
        self._active_mpo_state = ContextVar(f"circuit_resource_mpo_{suffix}", default=None)
        self._compression = ContextVar(f"circuit_resource_compression_{suffix}", default=None)
        self._tdvp_windows = weakref.WeakKeyDictionary()

    def __enter__(self) -> Self:
        """Install all experiment-local wrappers."""
        global _ACTIVE_TRACER
        if self._installed:
            msg = "This ResourceTracer is already active."
            raise RuntimeError(msg)
        if _ACTIVE_TRACER is not None:
            msg = "ResourceTracer monkeypatches are process-global; nested tracers are not supported."
            raise RuntimeError(msg)

        _ACTIVE_TRACER = self
        try:
            self._install()
        except BaseException:
            self._restore()
            _ACTIVE_TRACER = None
            raise
        self._installed = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Restore all patched callables, preserving any active exception."""
        del exc_type, exc_value, traceback
        global _ACTIVE_TRACER
        self._restore()
        self._installed = False
        _ACTIVE_TRACER = None
        return False

    @property
    def checkpoint_rows(self) -> list[dict[str, Any]]:
        """Return defensive copies of the recorded checkpoint rows."""
        return [dict(row) for row in self.rows]

    @property
    def peak_parameter_count(self) -> int:
        """Largest retained full-chain parameter count seen so far."""
        return max((int(row["parameter_count"]) for row in self.rows), default=0)

    @property
    def peak_bond_dim(self) -> int:
        """Largest retained virtual-bond dimension seen so far."""
        return max((int(row["peak_bond_dim"]) for row in self.rows), default=1)

    @contextmanager
    def gate_scope(self, **fields: Any) -> Iterator[None]:
        """Attach circuit-gate metadata to automatically generated checkpoints.

        Typical fields are ``model``, ``method``, ``chi_max``, ``step``,
        ``gate_index``, ``gate_name``, and ``sites``.  They are copied into every
        row generated while the scope is active.

        Args:
            **fields: Flat, JSON-compatible gate metadata.

        Yields:
            ``None`` while the physical gate is being applied.

        Raises:
            RuntimeError: If the tracer is inactive or gate scopes are nested.
            ValueError: If metadata attempts to replace a measured row field.
        """
        if not self._installed:
            msg = "gate_scope requires an active ResourceTracer context."
            raise RuntimeError(msg)
        if self._scope.get() is not None:
            msg = "Nested gate scopes are not supported."
            raise RuntimeError(msg)
        self._validate_fields(fields)
        scope = _GateScope(dict(fields))
        token = self._scope.set(scope)
        try:
            yield
        finally:
            self._scope.reset(token)

    def checkpoint(self, state: MPS, name: str, **fields: Any) -> dict[str, Any]:
        """Record an explicit full-chain checkpoint.

        This is intended for initial states and step endpoints; update-internal
        SVD checkpoints are generated automatically.

        Args:
            state: Full-chain MPS to measure.
            name: Descriptive checkpoint name such as ``"step_end"``.
            **fields: Additional flat metadata for this row.

        Returns:
            The newly appended row.
        """
        self._validate_fields(fields)
        return self._record_tensors(state.tensors, name, fields=fields)

    def _validate_fields(self, fields: Mapping[str, Any]) -> None:
        """Reject metadata that would shadow measured quantities."""
        overlap = sorted(_RESERVED_ROW_FIELDS.intersection(fields))
        if overlap:
            msg = f"Resource metadata uses reserved fields: {', '.join(overlap)}."
            raise ValueError(msg)

    def _record_tensors(
        self,
        tensors: Sequence[Any],
        checkpoint: str,
        *,
        fields: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one measured resource row from a full-chain tensor sequence."""
        parameter_count, peak_bond_dim, bonds = _tensor_resources(tensors)
        scope = self._scope.get()
        row = dict(scope.fields) if scope is not None else {}
        if fields is not None:
            row.update(fields)
        checkpoint_in_gate: int | None = None
        if scope is not None:
            checkpoint_in_gate = scope.checkpoint_count
            scope.checkpoint_count += 1
        row.update({
            "checkpoint": checkpoint,
            "checkpoint_index": len(self.rows),
            "checkpoint_in_gate": checkpoint_in_gate,
            "n_sites": len(tensors),
            "parameter_count": parameter_count,
            "peak_bond_dim": peak_bond_dim,
            "bond_dimensions": bonds,
        })
        self.rows.append(row)
        return row

    def _record_tdvp_split(self, short_state: Any, left_site: int, right_site: int) -> None:
        """Reconstruct the full parent chain and record one window-local split."""
        binding = self._tdvp_windows.get(short_state)
        if binding is None or self._scope.get() is None:
            return
        tensors = list(binding.parent.tensors)
        tensors[binding.first_site : binding.last_site + 1] = short_state.tensors
        self._record_tensors(
            tensors,
            "tdvp_split",
            fields={"updated_sites": [binding.first_site + left_site, binding.first_site + right_site]},
        )

    def _record_mpo_split(self, frame: _CompressionFrame, left: Any, right: Any) -> None:
        """Record one prospective MPS.compress output before assignment."""
        site = frame.next_site
        if site >= frame.state.length - 1:
            msg = "MPS.compress emitted more SVDs than its chain length permits."
            raise RuntimeError(msg)
        tensors = list(frame.state.tensors)
        tensors[site] = left
        tensors[site + 1] = right
        self._record_tensors(
            tensors,
            "mpo_compress_svd",
            fields={"updated_sites": [site, site + 1]},
        )
        frame.next_site += 1

    def _patch(self, owner: Any, attribute: str, replacement: Any) -> None:
        """Install one wrapper and remember its original callable."""
        original = getattr(owner, attribute)
        self._patches.append(_Patch(owner, attribute, original))
        setattr(owner, attribute, replacement)

    def _install(self) -> None:
        """Install TEBD, TDVP-window, and MPO-compression wrappers."""
        import mqt.yaqs.core.data_structures.mps as mps_module
        from mqt.yaqs.core.data_structures.mpo import MPO
        from mqt.yaqs.core.data_structures.mps import MPS
        from mqt.yaqs.digital import digital_tjm

        original_apply_window = digital_tjm.apply_window
        original_tebd = digital_tjm.apply_two_qubit_gate_tebd
        original_update_center = MPS.update_center_after_split
        original_compress = MPS.compress
        original_split_two_site = mps_module.split_two_site
        original_multiply_mps = MPO._multiply_mps

        def wrapped_apply_window(state: Any, mpo: Any, first_site: int, last_site: int, window_size: int) -> Any:
            short_state, short_mpo, window = original_apply_window(state, mpo, first_site, last_site, window_size)
            if self._scope.get() is not None:
                self._tdvp_windows[short_state] = _WindowBinding(state, int(window[0]), int(window[1]))
            return short_state, short_mpo, window

        def wrapped_tebd(state: Any, gate: Any, sim_params: Any) -> tuple[int, int]:
            result = original_tebd(state, gate, sim_params)
            if self._scope.get() is not None and abs(int(gate.sites[0]) - int(gate.sites[1])) == 1:
                left_site = min(int(gate.sites[0]), int(gate.sites[1]))
                self._record_tensors(
                    state.tensors,
                    "tebd_split",
                    fields={
                        "updated_sites": [left_site, left_site + 1],
                        "local_gate_name": str(gate.name),
                    },
                )
            return result

        def wrapped_update_center(
            state: Any,
            left_site: int,
            right_site: int,
            svd_distribution: str,
        ) -> None:
            original_update_center(state, left_site, right_site, svd_distribution)
            self._record_tdvp_split(state, int(left_site), int(right_site))

        def wrapped_multiply_mps(
            mpo: Any,
            state: Any,
            *,
            sim_params: Any,
            compress: bool,
        ) -> None:
            if self._scope.get() is None or not compress:
                original_multiply_mps(mpo, state, sim_params=sim_params, compress=compress)
                return
            token = self._active_mpo_state.set(state)
            try:
                original_multiply_mps(mpo, state, sim_params=sim_params, compress=compress)
            finally:
                self._active_mpo_state.reset(token)

        def wrapped_compress(
            state: Any,
            threshold: float,
            *,
            max_bond_dim: int | None = None,
            trunc_mode: str = "discarded_weight",
        ) -> None:
            if self._active_mpo_state.get() is not state or self._scope.get() is None:
                original_compress(
                    state,
                    threshold,
                    max_bond_dim=max_bond_dim,
                    trunc_mode=trunc_mode,
                )
                return

            self._record_tensors(state.tensors, "mpo_post_contraction")
            frame = _CompressionFrame(state)
            token = self._compression.set(frame)
            try:
                original_compress(
                    state,
                    threshold,
                    max_bond_dim=max_bond_dim,
                    trunc_mode=trunc_mode,
                )
            finally:
                self._compression.reset(token)
            expected_splits = max(state.length - 1, 0)
            if frame.next_site != expected_splits:
                msg = f"MPS.compress emitted {frame.next_site} SVDs; expected {expected_splits}."
                raise RuntimeError(msg)

        def wrapped_split_two_site(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
            left, right = original_split_two_site(*args, **kwargs)
            frame = self._compression.get()
            if frame is not None and self._scope.get() is not None:
                self._record_mpo_split(frame, left, right)
            return left, right

        self._patch(digital_tjm, "apply_window", wrapped_apply_window)
        self._patch(digital_tjm, "apply_two_qubit_gate_tebd", wrapped_tebd)
        self._patch(MPS, "update_center_after_split", wrapped_update_center)
        self._patch(MPO, "_multiply_mps", wrapped_multiply_mps)
        self._patch(MPS, "compress", wrapped_compress)
        self._patch(mps_module, "split_two_site", wrapped_split_two_site)

    def _restore(self) -> None:
        """Restore installed callables in reverse order."""
        while self._patches:
            patch = self._patches.pop()
            setattr(patch.owner, patch.attribute, patch.original)
        self._tdvp_windows.clear()
