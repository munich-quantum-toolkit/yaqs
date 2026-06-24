from __future__ import annotations

import inspect

import numpy as np

from mqt.yaqs.characterization.process_tensors.diagnostics.exact import ExactProbeProcess
from mqt.yaqs.characterization.process_tensors.diagnostics.probe import ProbeSet, probe_process


def _make_minimal_probe_set(*, cut: int = 1, k: int = 1, n_p: int = 2, n_f: int = 3) -> ProbeSet:
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return ProbeSet(
        cut=cut,
        k=k,
        past_features=np.zeros((n_p, cut, 32), dtype=np.float32),
        future_features=np.zeros((n_f, k - cut + 1, 32), dtype=np.float32),
        past_pairs=[[] for _ in range(n_p)],
        past_cut_meas=[z.copy() for _ in range(n_p)],
        future_prep_cut=[z.copy() for _ in range(n_f)],
        future_pairs=[[] for _ in range(n_f)],
    )


def test_probe_process_uses_object_backend() -> None:
    class DummyProcess:
        def evaluate_probe_set(self, probe_set: ProbeSet) -> np.ndarray:
            n_p = len(probe_set.past_pairs)
            n_f = len(probe_set.future_pairs)
            return np.zeros((n_p, n_f, 3), dtype=np.float32)

    out = probe_process(process=DummyProcess(), cut=1, k=1, n_pasts=2, n_futures=3, rng=np.random.default_rng(7))
    assert out["pauli_xyz_ij"].shape == (2, 3, 3)
    assert "entropy" in out


def test_exact_probe_process_hides_static_ctx_parameter() -> None:
    sig = inspect.signature(ExactProbeProcess.__init__)
    assert "static_ctx" not in sig.parameters
    assert "initial_psi" in sig.parameters


def test_exact_probe_process_builds_static_ctx_internally(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import mqt.yaqs.characterization.process_tensors.diagnostics.exact as exact_mod

    calls: dict[str, object] = {}

    def _fake_make_ctx(operator, sim_params, noise_model=None):  # noqa: ANN001
        calls["ctx_args"] = (operator, sim_params, noise_model)
        return "CTX"

    def _fake_simulate_sequences(**kwargs):  # noqa: ANN003
        calls["simulate_kwargs"] = kwargs
        n_tot = len(kwargs["psi_pairs_list"])
        return np.zeros((n_tot, 8), dtype=np.float32)

    monkeypatch.setattr(exact_mod, "make_mcwf_static_context", _fake_make_ctx)
    monkeypatch.setattr(exact_mod, "_simulate_sequences", _fake_simulate_sequences)

    class DummySimParams:
        dt = 0.1

    op = object()
    sim = DummySimParams()
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    process = ExactProbeProcess(operator=op, sim_params=sim, initial_psi=psi0, parallel=False)
    probe_set = _make_minimal_probe_set(cut=1, k=1, n_p=2, n_f=3)
    out = process.evaluate_probe_set(probe_set)

    assert out.shape == (2, 3, 3)
    assert calls["ctx_args"] == (op, sim, None)
    assert calls["simulate_kwargs"]["static_ctx"] == "CTX"
