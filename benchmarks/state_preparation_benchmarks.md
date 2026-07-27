# State-Preparation Benchmark Definition

This document defines the benchmark suite for comparing quantum state-preparation
methods against the same targets, ansatz family, and noise assumptions.

## Scope

- Qubit counts: `6` and `12`.
- Ansatz: brickwall circuit.
- Target states: 9 target-state families per qubit count.
- Noise families:
  1. Ballarin/Quantinuum angle-dependent local depolarizing noise from
     arXiv:2511.15674.
  2. Dephasing noise.
  3. Depolarizing noise.
- Primary metric: target-state fidelity after optimization.
- Secondary metrics: circuit depth, number of one-qubit gates, number of
  two-qubit gates, number of trainable parameters, optimization wall time, and
  number of noise trajectories or shots used for evaluation.

All methods should report both the noiseless fidelity and the noisy fidelity
under each selected noise configuration.

## Shared Circuit Ansatz

All benchmark methods should use the same brickwall ansatz family.

Required run metadata:

- `num_qubits`: `6` or `12`.
- `num_layers`: method-specific, but must be reported.
- `num_parameters`: total number of trainable parameters.
- `num_1q_gates`: total number of single-qubit gates in the final circuit.
- `num_2q_gates`: total number of two-qubit gates in the final circuit.
- `initialization`: random seed or warm-start rule.
- `optimizer`: optimizer name and hyperparameters.

## Noise Models

### Common Rates

Use the following rates for dephasing and depolarizing noise:

| Gate context | Strength |
| --- | ---: |
| Noise applied after single-qubit gates | `0.064%` (`6.4e-4`) |
| Jump operators applied after multi-qubit gates | `0.51%` (`5.1e-3`) |

The strength is interpreted as the per-gate jump/channel strength used by the
simulator. If a method uses a different internal parameterization, it must state
the conversion.

### N1: Ballarin/Quantinuum Angle-Dependent Local Depolarizing Noise

Source: Marco Ballarin, Juan Jose Garcia-Ripoll, David Hayes, and Michael
Lubasch, "Efficient quantum state preparation of multivariate functions using
tensor networks", arXiv:2511.15674, Eqs. (8)-(9).

Definition:

- Express each two-qubit entangler in the Quantinuum-native form
  `RZZ(theta) = exp(-i theta (Z ⊗ Z) / 2)`. The shared ansatz's `RXX` and
  `RYY` gates are compiled using noiseless single-qubit basis changes and one
  `RZZ` gate with the same angle. Any other two-qubit gate must be compiled to
  an explicit `RZZ` sequence, and the resulting angles and gate count must be
  reported.
- Angles use the YAQS/Qiskit radian convention. Canonicalize each native angle
  to `[-pi, pi)` and set `a = abs(theta)`. Using the magnitude is the benchmark
  convention that extends the paper's nonnegative calibration fit to signed
  parameterized rotations.
- Remove native two-qubit rotations with `a <= 1e-4` before counting gates or
  applying noise, matching the pruning rule used by Ballarin et al.
- Treat every single-qubit gate, including compilation basis changes, as
  noiseless.
- Immediately after each retained ideal `RZZ(theta)` on qubits `i` and `j`,
  apply the product channel `K_i(a) ⊗ K_j(a)`, where

```math
\epsilon(a) = 2.1 \times 10^{-4} + 1.43 \times 10^{-3} a,
```

```math
r(a) = \frac{1}{3}\left(1-\sqrt{1-\frac{5}{4}\epsilon(a)}\right),
```

```math
K_q(a)(\rho) =
  [1-3r(a)]\rho
  + r(a)\left(X_q\rho X_q + Y_q\rho Y_q + Z_q\rho Z_q\right).
```

An exact Kraus representation of each local channel is
`{sqrt(1-3r) I, sqrt(r) X, sqrt(r) Y, sqrt(r) Z}`. The two-qubit
channel therefore has the 16 tensor-product Kraus operators formed from the
two local sets. In a trajectory implementation, independently sample
`I`, `X`, `Y`, or `Z` on each participating qubit with probabilities
`1-3r`, `r`, `r`, and `r`, respectively, and apply the sampled Pauli pair
after the ideal gate. This is a product of two local depolarizing channels,
not a correlated two-qubit Pauli channel.

Post-gate placement and the use of `abs(theta)` for signed rotations are
explicit conventions of this benchmark; the paper does not specify those two
cases. Do not additionally apply the `6.4e-4` or `5.1e-3` common rates in this
configuration.

Benchmark identifier:

```text
noise=ballarin_coupled
```

The existing identifier is retained for compatibility; `coupled` means that
the two local channels are attached to the same native gate, not that their
Pauli samples are correlated.

### N2: Dephasing Noise

Dephasing should be benchmarked with explicit choices along two axes:

- Site support:
  - `single_site`: local `Z`-type dephasing on each affected qubit.
  - `two_site`: correlated `ZZ`-type dephasing on the two qubits of a
    multi-qubit gate.
- Gate placement:
  - `single_qubit_gates`: apply noise only after single-qubit gates.
  - `multi_qubit_gates`: apply noise only after multi-qubit gates.
  - `all_gates`: apply noise after single-qubit and multi-qubit gates.

Recommended configurations:

| Identifier | Site support | Gate placement | Strengths |
| --- | --- | --- | --- |
| `dephasing_1s_1q` | `single_site` | `single_qubit_gates` | `6.4e-4` after 1q gates |
| `dephasing_1s_2q` | `single_site` | `multi_qubit_gates` | `5.1e-3` after multi-qubit gates |
| `dephasing_1s_all` | `single_site` | `all_gates` | `6.4e-4` after 1q gates, `5.1e-3` after multi-qubit gates |
| `dephasing_2s_2q` | `two_site` | multi-qubit gates only | `5.1e-3` after multi-qubit gates |
| `dephasing_1s2s_all` | `single_site` plus `two_site` | `all_gates` | `6.4e-4` after 1q gates, `5.1e-3` for all jump operators after multi-qubit gates |

If the full benchmark matrix is too large, the minimum dephasing set is
`dephasing_1s_all` and `dephasing_2s_2q`.

### N3: Depolarizing Noise

Depolarizing should use the same support and placement axes as dephasing.

- Single-site depolarizing: local `X`, `Y`, and `Z` Pauli jump operators on each
  affected qubit.
- Two-site depolarizing/correlated Pauli noise: correlated two-qubit Pauli jump
  operators on the two qubits of a multi-qubit gate. The exact operator set must
  be documented by the implementation, for example `XX`, `YY`, `ZZ` only or all
  non-identity two-qubit Pauli products.

Recommended configurations:

| Identifier | Site support | Gate placement | Strengths |
| --- | --- | --- | --- |
| `depolarizing_1s_1q` | `single_site` | `single_qubit_gates` | `6.4e-4` after 1q gates |
| `depolarizing_1s_2q` | `single_site` | `multi_qubit_gates` | `5.1e-3` after multi-qubit gates |
| `depolarizing_1s_all` | `single_site` | `all_gates` | `6.4e-4` after 1q gates, `5.1e-3` after multi-qubit gates |
| `depolarizing_2s_2q` | `two_site` | multi-qubit gates only | `5.1e-3` after multi-qubit gates |
| `depolarizing_1s2s_all` | `single_site` plus `two_site` | `all_gates` | `6.4e-4` after 1q gates, `5.1e-3` for all jump operators after multi-qubit gates |

If the full benchmark matrix is too large, the minimum depolarizing set is
`depolarizing_1s_all` and `depolarizing_2s_2q`.

## Target States

Each target state is generated for both `n = 6` and `n = 12` qubits.
The final state vectors are stored in
`benchmarks/state_preparation_target_states.json` and can be regenerated with
`benchmarks/generate_state_preparation_targets.py`.

The JSON file encodes each complex amplitude as `[real, imaginary]`. Amplitude
index `k` uses little-endian computational-basis ordering, i.e., qubit `i` is
the `i`-th bit of `k`. The global phase is fixed so the largest-magnitude
amplitude is positive real.

The JSON records the NumPy and SciPy versions used to generate it as provenance.
The generator's `--check` mode ignores the exact version strings, compares all
benchmark-defining metadata exactly, and uses tight numerical tolerances only
for state amplitudes, norms, and TFIM ground-state energies.

### T1: Gaussian Amplitude State

Identifier:

```text
target=gaussian_mu0p5_sigma0p1
```

Definition:

- Grid: the endpoint-excluded quantics grid on `[0, 1)` from arXiv:2602.12042,
  Section 3.2.1. In the JSON little-endian basis convention, amplitude index
  `k` has qubit `i` equal to bit `i` of `k`, and qubit `i` has quantics weight
  `2^{-(i+1)}`:

```math
x_k = \sum_{i=0}^{n-1} \operatorname{bit}_i(k) 2^{-(i+1)}.
```

- Mean: `mu = 0.5`.
- Standard deviation: `sigma = 0.1`.
- The target encodes the classical Gaussian probability density `f(x)` as
  amplitudes `psi(x) = sqrt(f(x))`. The unnormalized amplitudes are therefore

```math
\psi(x) \propto \exp\left(-\frac{(x - 0.5)^2}{4(0.1)^2}\right).
```

- Normalize the amplitude vector to unit 2-norm.

### T2-T4: Transverse-Field Ising Model Ground States

Use ground states of the transverse-field Ising model (TFIM)

```math
H = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i.
```

Use the standard uniform open-chain 1D TFIM with `J = 1.0` and uniform
transverse field `h`. In the little-endian state-vector convention, site `i` is
qubit `i` and bit `i` of the computational-basis index. The benchmark includes
one state in each regime:

| Identifier | Regime | Condition | Eigensolver base seed |
| --- | --- | --- | --- |
| `tfim_ferro` | ferromagnetic | `h / J = 0.5` | `1729` |
| `tfim_critical` | critical | `h / J = 1.0` | `2718` |
| `tfim_para` | paramagnetic | `h / J = 1.5` | `3141` |

For an `n`-qubit target, the deterministic eigensolver initial-vector seed is
`base_seed + 10000 * n`. The base seed is not a physical disorder seed. The
generated JSON records the resulting initial-vector seed, uniform `J_i`,
uniform `h_i`, and ground-state energy for each qubit count.

### T5-T7: Dense Haar-Random States

Generate three complete random dense states for each qubit count.

| Identifier | Description | Seed |
| --- | --- | --- |
| `haar_random_1` | Dense normalized complex random state | `4001` |
| `haar_random_2` | Dense normalized complex random state | `4002` |
| `haar_random_3` | Dense normalized complex random state | `4003` |

Recommended generation rule:

1. Draw real and imaginary components independently from a standard normal
   distribution.
2. Normalize the resulting complex vector to unit 2-norm.
3. Store the seed and generator version in the run metadata.

### T8-T9: Random MPS States

Generate two random MPS states for each qubit count.

| Identifier | Bond dimension | Seed |
| --- | ---: | --- |
| `random_mps_bond2` | `2` | `5002` |
| `random_mps_bond3` | `3` | `5003` |

Recommended generation rule:

1. Draw real random MPS tensors with independent standard normal entries,
   matching Quimb's `qtn.MPS_rand_state(L=n, bond_dim=..., normalize=True)`
   default tensor distribution.
2. Normalize the resulting state.
3. Use the same generation rule for both `n = 6` and `n = 12`.
4. Report the open-boundary bond dimensions. Internal bonds use the requested
   bond dimension throughout, matching Quimb's convention.

## Benchmark Matrix

The base target matrix contains:

```text
2 qubit counts x 9 target states = 18 target instances
```

For each target instance, run:

1. Noiseless optimization and evaluation.
2. Ballarin-style coupled-noise evaluation.
3. Selected dephasing configurations.
4. Selected depolarizing configurations.

Each reported result must include the exact noise identifier, target identifier,
qubit count, circuit size, optimizer settings, training budget, and test
evaluation budget.

## Reporting Template

Use one row per final benchmark result. The fidelity columns distinguish the
optimization trajectories from the independent test evaluation.

| Field | Description |
| --- | --- |
| `method` | Name of the state-preparation method |
| `num_qubits` | `6` or `12` |
| `target_id` | One of the target identifiers above |
| `noise_id` | Noise configuration identifier |
| `seed` | Target-generation seed or `none` |
| `ansatz` | Brickwall ansatz details |
| `num_layers` | Final number of ansatz layers |
| `num_parameters` | Final number of trainable parameters |
| `num_1q_gates` | Final one-qubit gate count |
| `num_2q_gates` | Final two-qubit gate count |
| `optimizer_budget` | Iterations, trajectories, shots, or other optimization budget |
| `train_trajectories_or_shots` | Trajectories or shots used during optimization |
| `train_fidelity` | Final fidelity after optimization on the same trajectories or shots used for training |
| `test_noiseless_fidelity` | Final trained parameters evaluated without noise |
| `test_noisy_fidelity` | Final trained parameters evaluated under `noise_id` with fresh trajectories or shots |
| `test_trajectories_or_shots` | Fresh trajectories or shots used for the test evaluation |
| `wall_time_seconds` | End-to-end runtime |
| `notes` | Deviations, failures, or implementation details |

## Open Item To Freeze

- Exact two-site depolarizing operator set.
