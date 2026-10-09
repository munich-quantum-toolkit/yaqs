# MQT YAQS — Simulation and characterization of quantum systems and their environments

MQT YAQS (pronounced "yaks" like the animals) is a Python library designed for
**scalable, computationally efficient** simulation and characterization of open
quantum dynamics, noisy quantum circuits, and hardware-realistic device models.
MQT YAQS applies state-of-the-art techniques in these areas—parallelized
trajectories, tensor-network compression, and backends matched to problem
size—wherever possible (see {doc}`references`). It is developed as part of the
[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io) by the
[Chair for Design Automation](https://www.cda.cit.tum.de/) at the
[Technical University of Munich](https://www.tum.de).

This documentation provides a comprehensive guide to the MQT YAQS library,
including {doc}`installation instructions <installation>`, notebook-like
examples, and detailed {doc}`API documentation <api/mqt/yaqs/index>`. The source
code of MQT YAQS is publicly available on GitHub at
[munich-quantum-toolkit/yaqs](https://github.com/munich-quantum-toolkit/yaqs),
while pre-built binaries are available via
[PyPI](https://pypi.org/project/mqt.yaqs/) for all major operating systems and
all modern Python versions.

```{toctree}
:hidden:

self
```

## User guide

MQT YAQS targets workloads that need **scale and efficiency**: large noisy
circuits, long analog time evolution, and hardware models with many degrees of
freedom. For smaller systems, **MCWF** (`vector`) and **Lindblad**
(`density_matrix`) analog backends are available as well; see
{doc}`examples/representation_comparison`.

The pages below are **executable notebooks**: code cells run during the
documentation build, so examples stay in sync with the library. New users should
start with {doc}`installation`, then {doc}`examples/quickstart`.

```{mermaid}
flowchart LR
  state[State]
  op[Hamiltonian or QuantumCircuit]
  params["AnalogSimParams / DigitalSimParams"]
  sim[Simulator]
  result[Result]
  state --> sim
  op --> sim
  params --> sim
  sim --> result
```

### Find a guide

| I want to…                                                                 | Read                                                                        |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Run my first simulation in under a minute                                  | {doc}`examples/quickstart`                                                  |
| Configure truncation, presets, and trajectories                            | {doc}`examples/simulation_parameters`                                       |
| Build Hamiltonians (Pauli, Hubbard, transmon, trapped ion, …)              | {doc}`examples/hamiltonians`                                                |
| Simulate open-system (analog) dynamics with noise                          | {doc}`examples/analog_simulation`                                           |
| Model realistic noise (log-normal and other distributions)                 | {doc}`examples/realistic_noise_models`                                      |
| Define custom single-site jump operators                                   | {doc}`examples/realistic_noise_models`                                      |
| Compare scalable MPS, MCWF, and Lindblad analog paths                      | {doc}`examples/representation_comparison`                                   |
| Two-time correlations and typicality ensembles                             | {doc}`examples/ensemble_evolution`                                          |
| Scheduled jumps at fixed times                                             | {doc}`examples/scheduled_jumps`                                             |
| Transfer an excitation between superconducting qubits                      | {doc}`examples/transmon_emulation`                                          |
| Transport a trapped ion and study motional noise                           | {doc}`examples/trapped_ion`                                                 |
| Characterize environmental memory effects via probing the process          | {doc}`examples/characterization`                                            |
| Study how long environmental memory persists in a system                   | {ref}`Memory persistence <reset-delay>` in {doc}`examples/characterization` |
| Train a surrogate and predict how a system evolves under control sequences | {doc}`examples/memory_surrogate`                                            |
| Build a Markovian noise digital twin from measured trajectories            | {doc}`examples/digital_twin`                                                |
| Validate predictions at short temporal horizons with exact references      | {doc}`examples/memory_surrogate`                                            |
| Simulate a circuit and read observables                                    | {doc}`examples/circuit_observables`                                         |
| Get hardware-like shot histograms                                          | {doc}`examples/circuit_shots`                                               |
| Combine analog evolution and digital operations in one program             | {doc}`examples/digital_analog_simulation`                                   |
| Verify two circuits are equivalent                                         | {doc}`examples/equivalence_checking`                                        |
| Custom gate translation                                                    | {doc}`examples/custom_gates`                                                |

```{toctree}
:caption: Start here
:hidden:
:maxdepth: 1
:titlesonly:

Installation <installation>
Quickstart <examples/quickstart>
```

```{toctree}
:caption: Simulation setup
:hidden:
:maxdepth: 1
:titlesonly:

Quantum states <examples/state_initialization>
Hamiltonians <examples/hamiltonians>
Noise models <examples/realistic_noise_models>
State representations <examples/representation_comparison>
Simulation parameters <examples/simulation_parameters>
Simulator configuration <examples/simulator_initialization>
```

```{toctree}
:caption: Simulation workflows
:hidden:
:maxdepth: 1
:titlesonly:

Analog simulation <examples/analog_simulation>
Digital (circuit) simulation <examples/circuit_observables>
Shot-based simulation <examples/circuit_shots>
Analog-digital simulation <examples/digital_analog_simulation>
```

```{toctree}
:caption: Characterization and verification
:hidden:
:maxdepth: 1
:titlesonly:

Environmental memory characterization <examples/characterization>
Noise model characterization <examples/digital_twin>
Circuit verification <examples/equivalence_checking>
```

```{toctree}
:caption: Advanced examples
:hidden:
:maxdepth: 1
:titlesonly:

Ensemble evolution <examples/ensemble_evolution>
Scheduled jumps <examples/scheduled_jumps>
Custom gates <examples/custom_gates>
Superconducting qubit (transmon) emulation <examples/transmon_emulation>
Trapped ion emulation <examples/trapped_ion>
Non-Markovian transformer models (experimental) <examples/memory_surrogate>

```

```{toctree}
:caption: Reference and contributing
:hidden:
:maxdepth: 1
:titlesonly:

API reference <api/mqt/yaqs/index>
References and citations <references>
Changelog <CHANGELOG>
Upgrade guide <UPGRADING>
Contributing <contributing>
AI usage <ai_usage>
Development tools <tooling>
Support <support>
```

## Contributors and supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by
the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the
[Technical University of Munich](https://www.tum.de/) and supported by
[MQSC](https://mq.sc). Among others, it is part of the
[Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss)
ecosystem, which is being developed as part of the
[Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

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

The MQT will remain free, open-source, and permissively licensed—now and in the
future. We are firmly committed to keeping it open and actively maintained for
the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories:
  <https://github.com/munich-quantum-toolkit>
- Contributing code, documentation, tests, or examples via issues and pull
  requests
- Citing the methods and software you use in your publications (see
  {doc}`References <references>`)
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: <https://github.com/sponsors/munich-quantum-toolkit>

<p align="center">
<iframe src="https://github.com/sponsors/munich-quantum-toolkit/button" title="Sponsor munich-quantum-toolkit" height="32" width="114" style="border: 0; border-radius: 6px;"></iframe>
</p>
