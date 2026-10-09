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

Start with installation and the quickstart, then choose a guide for your task.
The examples include working code and plots.

| Section                           | Guides                                                                                                                                                                                                                                                                                                                                                               |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Start here                        | {doc}`Installation <installation>` · {doc}`Quickstart <examples/quickstart>`                                                                                                                                                                                                                                                                                         |
| Simulation setup                  | {doc}`Quantum states <examples/state_initialization>` · {doc}`Hamiltonians <examples/hamiltonians>` · {doc}`Noise models <examples/realistic_noise_models>` · {doc}`State representations <examples/representation_comparison>` · {doc}`Simulation parameters <examples/simulation_parameters>` · {doc}`Simulator configuration <examples/simulator_initialization>` |
| Simulation workflows              | {doc}`Analog <examples/analog_simulation>` · {doc}`Digital circuits <examples/circuit_observables>` · {doc}`Shot-based circuits <examples/circuit_shots>` · {doc}`Analog-digital <examples/digital_analog_simulation>`                                                                                                                                               |
| Emulation                         | {doc}`Superconducting qubits <examples/transmon_emulation>` · {doc}`Trapped ions <examples/trapped_ion>`                                                                                                                                                                                                                                                             |
| Characterization and verification | {doc}`Environmental memory <examples/characterization>` · {doc}`Noise characterization <examples/digital_twin>` · {doc}`Circuit verification <examples/equivalence_checking>`                                                                                                                                                                                        |
| Advanced examples                 | {doc}`Ensemble evolution <examples/ensemble_evolution>` · {doc}`Non-Markovian surrogate models (experimental) <examples/memory_surrogate>`                                                                                                                                                                                                                           |
| Reference and contributing        | {doc}`API <api/mqt/yaqs/index>` · {doc}`Citations <references>` · {doc}`Changelog <CHANGELOG>` · {doc}`Upgrade guide <UPGRADING>` · {doc}`Contributing <contributing>` · {doc}`Support <support>`                                                                                                                                                                    |

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
:caption: Emulation
:hidden:
:maxdepth: 1
:titlesonly:

Superconducting qubit (transmon) emulation <examples/transmon_emulation>
Trapped ion emulation <examples/trapped_ion>
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
