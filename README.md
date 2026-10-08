[![PyPI](https://img.shields.io/pypi/v/mqt.yaqs?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.yaqs/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos%20%7C%20windows-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/yaqs/ci.yml?branch=main&style=flat-square&logo=github&label=ci)](https://github.com/munich-quantum-toolkit/yaqs/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/yaqs/cd.yml?style=flat-square&logo=github&label=cd)](https://github.com/munich-quantum-toolkit/yaqs/actions/workflows/cd.yml)
[![Documentation](https://img.shields.io/readthedocs/mqt-yaqs?logo=readthedocs&style=flat-square)](https://mqt.readthedocs.io/projects/yaqs)
[![codecov](https://img.shields.io/codecov/c/github/munich-quantum-toolkit/yaqs?style=flat-square&logo=codecov)](https://codecov.io/gh/munich-quantum-toolkit/yaqs)

<p align="center">
  <a href="https://mqt.readthedocs.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-dark.svg" width="60%">
      <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-light.svg" width="60%" alt="MQT Logo">
    </picture>
  </a>
</p>

# MQT YAQS — Simulation and characterization of quantum systems and their environments

MQT YAQS (pronounced "yaks") is a Python library for simulating quantum systems
and studying their interaction with the environment. It supports analog
evolution, noisy quantum circuits, circuit equivalence checking, and
characterization of environmental memory and noise models. It is part of the
[_Munich Quantum Toolkit (MQT)_](https://mqt.readthedocs.io).

YAQS uses tensor networks and quantum trajectories, with statevector and
density-matrix backends for smaller analog simulations. The cost of
tensor-network simulation depends on entanglement and accuracy settings.

<p align="center">
  <a href="https://mqt.readthedocs.io/projects/yaqs">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

## Key Features

| Capability                                                                                                              | What users can do                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Analog simulation](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/analog_simulation.html)                 | Simulate closed and open quantum dynamics.                                                                                                                |
| [Digital simulation](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/circuit_observables.html)              | Simulate noisy circuits, measure observables, and sample shots.                                                                                           |
| [Digital–analog simulation](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/digital_analog_simulation.html) | Combine analog evolution and circuit operations.                                                                                                          |
| [Equivalence checking](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/equivalence_checking.html)           | Compare quantum circuits.                                                                                                                                 |
| [Environmental memory](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/characterization.html)               | Probe memory through intervention sequences and construct process tensors.                                                                                |
| [Surrogate models](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/memory_surrogate.html)                   | Train models to predict probe dynamics under interventions; requires PyTorch.                                                                             |
| [Noise characterization](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/digital_twin.html)                 | Fit Lindblad jump rates from observable time series.                                                                                                      |
| [Hardware modeling](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/hamiltonians.html)                      | Build device Hamiltonians and use [distributed noise strengths](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/realistic_noise_models.html). |

If you have any questions, feel free to create a
[discussion](https://github.com/munich-quantum-toolkit/yaqs/discussions) or an
[issue](https://github.com/munich-quantum-toolkit/yaqs/issues) on
[GitHub](https://github.com/munich-quantum-toolkit/yaqs).

## Getting Started

MQT YAQS requires **Python 3.11 or newer** and runs on Linux, macOS, and
Windows. Install it from [PyPI](https://pypi.org/project/mqt.yaqs/) in a virtual
environment using `uv`:

```console
uv pip install mqt.yaqs
```

Or use `pip`:

```console
python -m pip install mqt.yaqs
```

See the
[installation guide](https://mqt.readthedocs.io/projects/yaqs/en/latest/installation.html)
for environment setup and development installation.

Optional extras support
[OpenQASM 3 input](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/circuit_observables.html)
(`qasm3`) and
[surrogate training and prediction](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/memory_surrogate.html)
(`torch`). The linked guides include installation steps.

### Noisy analog evolution

Evolve a 50-site Ising chain with local damping and print the final mean ⟨Z₀⟩.

```python
from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State

if __name__ == "__main__":
    length = 50
    state = State(length, initial="zeros")
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
    noise = NoiseModel([{"name": "lowering", "sites": [site], "strength": 0.05} for site in range(length)])
    params = AnalogSimParams(
        observables=[Observable("z", sites=0)],
        elapsed_time=1.0,
        dt=0.1,
    )
    simulator = Simulator()
    result = simulator.run(state, hamiltonian, params, noise)
    print(f"Final mean <Z_0>: {result.expectation_values[0][-1]:.3f}")
```

[analog simulation guide](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/analog_simulation.html)

### Digital circuit and shot readout

Prepare a 50-qubit GHZ state and print sampled bitstring counts.

```python
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import DigitalSimParams, Simulator, State

if __name__ == "__main__":
    length = 50
    state = State(length, initial="zeros")
    circuit = QuantumCircuit(length)
    circuit.h(0)
    for site in range(1, length):
        circuit.cx(site - 1, site)
    circuit.measure_all()

    params = DigitalSimParams(shots=1024)
    simulator = Simulator()
    result = simulator.run(state, circuit, params)
    print({format(outcome, f"0{length}b"): count for outcome, count in result.counts.items()})
```

[shot-readout guide](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/circuit_shots.html)

### Environmental memory characterization

Probe environmental memory in a 50-site chain and print the memory diagnostics.

```python
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

if __name__ == "__main__":
    hamiltonian = Hamiltonian.ising(length=50, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1)
    characterizer = MemoryCharacterizer()
    result = characterizer.characterize(
        hamiltonian,
        params,
        num_interventions=6,
        preset="quick",
    )
    print(result.summary())
```

[memory characterization guide](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/characterization.html)

Trajectory count, time-step size, and MPS truncation affect numerical accuracy;
see the
[simulation parameter guide](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/simulation_parameters.html).

The examples use default parallel execution. See the
[execution guide](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/simulator_initialization.html)
for worker controls.

**Documentation:**
[Quickstart](https://mqt.readthedocs.io/projects/yaqs/en/latest/examples/quickstart.html)
·
[API reference](https://mqt.readthedocs.io/projects/yaqs/en/latest/api/mqt/yaqs/index.html)
·
[Full documentation](https://mqt.readthedocs.io/projects/yaqs)

## Cite This

Please cite the work that best fits your use case.

### Peer-Reviewed Research

When citing the underlying methods and research, please reference the most
relevant peer-reviewed publications from the list below:

[[1]](https://www.nature.com/articles/s41467-025-66846-x) A. Sander, M.
Fröhlich, M. Eigel, J. Eisert, P. Gelß, M. Hintermüller, R. M. Milbradt, R.
Wille, C. B. Mendl. Large-scale stochastic simulation of open quantum systems.
_Nature Communications_ _16_, 11074 (2025).

[[2]](https://journals.aps.org/prresearch/abstract/10.1103/3q71-y8cf) A. Sander,
L. Burgholzer, and R. Wille. Equivalence checking of quantum circuits via
intermediary matrix product operator. _Phys. Rev. Research_ _7_, 023261 (2025).

[[3]](https://arxiv.org/abs/2508.10096) A. Sander, M. Fröhlich, M. Ali, M.
Eigel, J. Eisert, M. Hintermüller, C. B. Mendl, R. M. Milbradt, R. Wille.
Quantum circuit simulation with a local time-dependent variational principle.
_arXiv:2508.10096 (2025)._

[[4]](https://arxiv.org/abs/2606.13779) A. Sander, S. Cichy, M. Eigel, J.
Eisert, M. Fröhlich, T. Peham, R. Wille. Computational regimes in
matrix-product-state-based quantum trajectory simulations.
_arXiv:2606.13779 (2026)._

### The Munich Quantum Toolkit (the project)

When discussing the overall MQT project or its ecosystem, cite the MQT Handbook:

```bibtex
@inproceedings{mqt,
  title        = {The {{MQT}} Handbook: {{A}} Summary of Design Automation Tools and Software for Quantum Computing},
  shorttitle   = {{The MQT Handbook}},
  author       = {Wille, Robert and Berent, Lucas and Forster, Tobias and Kunasaikaran, Jagatheesan and Mato, Kevin and Peham, Tom and Quetschlich, Nils and Rovara, Damian and Sander, Aaron and Schmid, Ludwig and Schoenberger, Daniel and Stade, Yannick and Burgholzer, Lukas},
  year         = 2024,
  booktitle    = {IEEE International Conference on Quantum Software (QSW)},
  doi          = {10.1109/QSW62656.2024.00013},
  eprint       = {2405.17543},
  eprinttype   = {arxiv},
  addendum     = {A live version of this document is available at \url{https://mqt.readthedocs.io}}
}
```

## Contributors and Supporters

<p align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/yaqs/main/images/banner.jpeg" width="40%" alt="MQT YAQS artwork">
</p>

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by
the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the
[Technical University of Munich](https://www.tum.de/) and supported by
[MQSC](https://mq.sc). Among others, it is part of the
[Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss)
ecosystem, which is being developed as part of the
[Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Partner Logos">
  </picture>
</p>

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
- Citing the MQT in your publications (see [Cite This](#cite-this))
- Citing our research in your publications (see
  [References](https://mqt.readthedocs.io/projects/yaqs/en/latest/references.html))
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: <https://github.com/sponsors/munich-quantum-toolkit>

<p align="center">
  <a href="https://github.com/sponsors/munich-quantum-toolkit">
  <img width=20% src="https://img.shields.io/badge/Sponsor-white?style=for-the-badge&logo=githubsponsors&labelColor=black&color=blue" alt="Sponsor the MQT" />
  </a>
</p>

---

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European Research Council
(ERC) under the European Union's Horizon 2020 research and innovation program
(grant agreement No. 101001318), the Bavarian State Ministry for Science and
Arts through the Distinguished Professorship Program, as well as the Munich
Quantum Valley, which is supported by the Bavarian state government with funds
from the Hightech Agenda Bayern Plus.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>
