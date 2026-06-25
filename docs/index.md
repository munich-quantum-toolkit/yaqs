# MQT YAQS — Scalable simulation for open systems, noisy circuits, and realistic hardware

```{only} html
[![PyPI](https://img.shields.io/pypi/v/mqt.yaqs?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.yaqs/)
[![CI](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/yaqs/ci.yml?branch=main&style=flat-square&logo=github&label=ci)](https://github.com/munich-quantum-toolkit/yaqs/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/readthedocs/mqt-yaqs?logo=readthedocs&style=flat-square)](https://mqt.readthedocs.io/projects/yaqs)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
```

```{raw} latex
\begin{abstract}
```

YAQS (pronounced "yaks" like the animals) is a Python library designed for **scalable, computationally efficient** simulation of open quantum dynamics, noisy quantum circuits, and hardware-realistic device models. YAQS applies state-of-the-art techniques in these areas—parallelized trajectories, tensor-network compression, and backends matched to problem size—wherever possible (see {doc}`references`).
It is developed as part of the [Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io) by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de).

This documentation provides a comprehensive guide to the MQT YAQS library, including {doc}`installation instructions <installation>`, notebook-like examples, and detailed {doc}`API documentation <api/mqt/yaqs/index>`.
The source code of MQT YAQS is publicly available on GitHub at [munich-quantum-toolkit/yaqs](https://github.com/munich-quantum-toolkit/yaqs), while pre-built binaries are available via [PyPI](https://pypi.org/project/mqt.yaqs/) for all major operating systems and all modern Python versions.

````{only} latex
```{note}
A live version of this document is available at [mqt.readthedocs.io/projects/yaqs](https://mqt.readthedocs.io/projects/yaqs).
```
````

```{raw} latex
\end{abstract}

\sphinxtableofcontents
```

```{toctree}
:hidden:

self
```

## User guide

YAQS targets workloads that need **scale and efficiency**: large noisy circuits, long analog time evolution, and hardware models with many degrees of freedom. For smaller systems, **MCWF** (`vector`) and **Lindblad** (`density_matrix`) analog backends are available as well; see {doc}`examples/representation_comparison`.

The pages below are **executable notebooks**: code cells run during the documentation build, so examples stay in sync with the library. New users should start with {doc}`installation`, then {doc}`examples/quickstart`.

```{mermaid}
flowchart LR
  state[State]
  op[Hamiltonian or QuantumCircuit]
  params["AnalogSimParams / StrongSimParams / WeakSimParams"]
  sim[Simulator]
  result[Result]
  state --> sim
  op --> sim
  params --> sim
  sim --> result
```

### Learning paths

| I want to…                                                    | Read                                       |
| ------------------------------------------------------------- | ------------------------------------------ |
| Run my first simulation in under a minute                     | {doc}`examples/quickstart`                 |
| Configure truncation, presets, and trajectories               | {doc}`examples/simulation_parameters`      |
| Build Hamiltonians (Pauli, Hubbard, transmon, trapped ion, …) | {doc}`examples/hamiltonians`               |
| Simulate open-system (analog) dynamics with noise             | {doc}`examples/analog_simulation`          |
| Model realistic noise (log-normal and other distributions)    | {doc}`examples/realistic_noise_models`     |
| Define custom single-site jump operators                      | {doc}`examples/realistic_noise_models` § 6 |
| Compare scalable MPS, MCWF, and Lindblad analog paths         | {doc}`examples/representation_comparison`  |
| Simulate a noisy circuit and read observables                 | {doc}`examples/circuit_simulation`         |
| Get hardware-like shot histograms                             | {doc}`examples/weak_circuit_simulation`    |
| Verify two circuits are equivalent                            | {doc}`examples/equivalence_checking`       |
| Two-time correlations and typicality ensembles                | {doc}`examples/ensemble_evolution`         |
| Scheduled jumps at fixed times                                | {doc}`examples/scheduled_jumps`            |
| Transmon–resonator SWAP (noiseless vs noisy)                  | {doc}`examples/transmon_emulation`         |
| Trapped-ion position-grid dynamics                            | {doc}`examples/trapped_ion`                |
| Custom gate translation                                       | {doc}`examples/custom_gates`               |

### Characterization

| I want to…                                   | Read                                      |
| -------------------------------------------- | ----------------------------------------- |
| Characterize operational memory (start here) | {doc}`examples/characterization`          |
| Ground-truth S_V from simulator rollouts     | {doc}`examples/characterization`          |
| Fast repeated memory metrics via surrogate   | {doc}`examples/characterization`          |
| Tune surrogate training                      | {doc}`examples/process_tensor_surrogates` |
| V-matrix theory and validation               | {doc}`examples/operational_memory`        |
| Reference exact combs (small `k` only)       | {doc}`examples/reference_exact_combs`     |

```{toctree}
:caption: Getting started
:hidden:
:maxdepth: 1
:titlesonly:

installation
examples/quickstart
examples/state_initialization
examples/simulator_initialization
examples/simulation_parameters
```

```{toctree}
:caption: Analog simulation
:hidden:
:maxdepth: 1
:titlesonly:

examples/hamiltonians
examples/analog_simulation
examples/realistic_noise_models
examples/scheduled_jumps
examples/ensemble_evolution
examples/representation_comparison
examples/transmon_emulation
examples/trapped_ion
```

## Characterization

Characterize **non-Markovian memory** in open quantum processes via split-cut V-matrix
diagnostics (`S_V`, rank, spectrum). Start with {doc}`examples/characterization` for
`MemoryCharacterizer` workflows (exact simulator and surrogate paths).

```{toctree}
:caption: Characterization
:hidden:
:maxdepth: 1
:titlesonly:

examples/characterization
examples/process_tensor_surrogates
examples/operational_memory
examples/reference_exact_combs
```

```{toctree}
:caption: Digital circuits
:hidden:
:maxdepth: 1
:titlesonly:

examples/circuit_simulation
examples/weak_circuit_simulation
examples/custom_gates
examples/equivalence_checking
```

```{toctree}
:caption: Reference
:hidden:
:maxdepth: 1
:titlesonly:

references
CHANGELOG
UPGRADING
```

````{only} not latex
```{toctree}
:caption: Developers
:hidden:
:maxdepth: 1
:titlesonly:

contributing
ai_usage
tooling
support
```
````

```{toctree}
:caption: API Reference
:hidden:
:glob:
:maxdepth: 1

api/mqt/yaqs/index
```

```{only} html
## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by [MQSC](https://mq.sc).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<div style="margin-top: 0.5em">
<div class="only-light" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Banner">
</div>
<div class="only-dark" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%" alt="MQT Banner">
</div>
</div>

Thank you to all the contributors who have helped make MQT YAQS a reality!

<p align="center">
<a href="https://github.com/munich-quantum-toolkit/yaqs/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/yaqs" />
</a>
</p>

The MQT will remain free, open-source, and permissively licensed—now and in the future.
We are firmly committed to keeping it open and actively maintained for the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories: [https://github.com/munich-quantum-toolkit](https://github.com/munich-quantum-toolkit)
- Contributing code, documentation, tests, or examples via issues and pull requests
- Citing the MQT in your publications (see {doc}`References <references>`)
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: [https://github.com/sponsors/munich-quantum-toolkit](https://github.com/sponsors/munich-quantum-toolkit)

<p align="center">
<iframe src="https://github.com/sponsors/munich-quantum-toolkit/button" title="Sponsor munich-quantum-toolkit" height="32" width="114" style="border: 0; border-radius: 6px;"></iframe>
</p>
```
