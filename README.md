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
      <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/yaqs/main/images/banner.jpeg" width="60%" alt="MQT YAQS Banner">
    </picture>
  </a>
</p>

# MQT YAQS - A Tool for Simulating Open Quantum Systems, Noisy Quantum Circuits, and Realistic Quantum Hardware

MQT YAQS (pronounced "yaks" like the animals) is a Python library, primarily focused on simulating open quantum systems, noisy quantum circuits, and designing realistic quantum hardware.
It is part of the [_Munich Quantum Toolkit (MQT)_](https://mqt.readthedocs.io).

<p align="center">
  <a href="https://mqt.readthedocs.io/projects/yaqs">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

## Key Features

- **Simulation of Open Quantum Systems (Analog Simulation)**: Simulate large-scale open quantum systems with a parallelized implementation using the Tensor Jump Method (TJM) [2]
- **Noisy Quantum Circuit Simulation (Digital Simulation)**: Investigate the effect of noise on large quantum circuits [3]
- **Equivalence Checking of Quantum Circuits**: Check the equivalence or non-equivalence of quantum circuits with a scalable MPO-based method [1]
- **WIP: Quantum Hardware Design**: Design better quantum hardware with realistic simulation methods

If you have any questions, feel free to create a [discussion](https://github.com/munich-quantum-toolkit/yaqs/discussions) or an [issue](https://github.com/munich-quantum-toolkit/yaqs/issues) on [GitHub](https://github.com/munich-quantum-toolkit/yaqs).

## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

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

The MQT will remain free, open-source, and permissively licensed—now and in the future.
We are firmly committed to keeping it open and actively maintained for the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories: https://github.com/munich-quantum-toolkit
- Contributing code, documentation, tests, or examples via issues and pull requests
- Citing the MQT in your publications (see [Cite This](#cite-this))
- Citing our research in your publications (see [References](https://mqt.readthedocs.io/projects/yaqs/en/latest/references.html))
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: https://github.com/sponsors/munich-quantum-toolkit

<p align="center">
  <a href="https://github.com/sponsors/munich-quantum-toolkit">
  <img width=20% src="https://img.shields.io/badge/Sponsor-white?style=for-the-badge&logo=githubsponsors&labelColor=black&color=blue" alt="Sponsor the MQT" />
  </a>
</p>

## Getting Started

`mqt.yaqs` is available via [PyPI](https://pypi.org/project/mqt.yaqs/).

```console
(.venv) $ pip install mqt.yaqs
```

**Detailed documentation and examples are available at [ReadTheDocs](https://mqt.readthedocs.io/projects/yaqs).**

## System Requirements

MQT YAQS can be installed on all major operating systems with all [officially supported Python versions](https://devguide.python.org/versions/).
Building (and running) is continuously tested under Linux, macOS, and Windows using the [latest available system versions for GitHub Actions](https://github.com/actions/runner-images).

## Cite This

Please cite the work that best fits your use case.

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

### Peer-Reviewed Research

When citing the underlying methods and research, please reference the most relevant peer-reviewed publications from the list below:

[[1]](https://arxiv.org/pdf/2410.10946)
A. Sander, L. Burgholzer, and R. Wille.
Equivalence checking of quantum circuits via intermediary matrix product operator.
_Phys. Rev. Research 7, 023261_, 2023.

[[2]](https://arxiv.org/pdf/2501.17913)
A. Sander, M. Fröhlich, M. Eigel, J. Eisert, P. Gelß, M. Hintermüller, R. M. Milbradt, R. Wille, C. B. Mendl.
Large-scale stochastic simulation of open quantum systems.

[[3]](https://arxiv.org/abs/2508.10096)
A. Sander, M. Fröhlich, M. Ali, M. Eigel, J. Eisert, M. Hintermüller, C. B. Mendl, R. M. Milbradt, R. Wille
Quantum circuit simulation with a local time-dependent variational principle.

---

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 101001318), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as well as the Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>
