[![PyPI](https://img.shields.io/pypi/v/mqt.yaqs?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.yaqs/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos%20%7C%20windows-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/cda-tum/mqt-yaqs/ci.yml?branch=main&style=flat-square&logo=github&label=ci)](https://github.com/cda-tum/mqt-yaqs/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/cda-tum/mqt-yaqs/cd.yml?style=flat-square&logo=github&label=cd)](https://github.com/cda-tum/mqt-yaqs/actions/workflows/cd.yml)
[![codecov](https://img.shields.io/codecov/c/github/cda-tum/mqt-yaqs?style=flat-square&logo=codecov)](https://codecov.io/gh/cda-tum/mqt-yaqs)

![Banner](images/banner.jpeg)

# MQT YAQS: A Tool for Simulating Open Quantum Systems, Noisy Quantum Circuits, and Realistic Quantum Hardware

YAQS (pronounced "yaks" like the animals) is a Python library, primarily focused on simulating open quantum systems, noisy quantum circuits, and designing realistic quantum hardware.

YAQS is part of the [_Munich Quantum Toolkit_ (_MQT_)](https://mqt.readthedocs.io) developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/).

<p align="center">
  <a href="https://mqt.readthedocs.io/projects/yaqs">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

If you have any questions, feel free to create a [discussion](https://github.com/cda-tum/mqt-yaqs/discussions) or an [issue](https://github.com/cda-tum/mqt-yaqs/issues) on [GitHub](https://github.com/cda-tum/mqt-yaqs).

## Features

- **Simulation of Open Quantum Systems**: Simulate large-scale open quantum systems with a paralellized implementation using the Tensor Jump Method (TJM) ([Large-scale stochastic simulation of open quantum systems](https://arxiv.org/abs/2501.17913v1))
- **Equivalence Checking of Quantum Circuits**: Check the equivalence or non-equivalence of quantum circuits with a scalable MPO-based method ([Equivalence Checking of Quantum Circuits via Intermediary Matrix Product Operator](https://arxiv.org/abs/2410.10946))
- **Noisy Quantum Circuit Simulation**: Investigate the effect of noise on large quantum circuits

**Detailed documentation and examples are available at [ReadTheDocs](https://mqt.readthedocs.io/projects/yaqs).**

## Upcoming Features (Check back soon)

- **WIP: Quantum Hardware Design**: Design better quantum hardware with realistic simulation methods

---

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European
Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement
No. 101001318), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as well as the
Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/tum_dark.svg" width="28%">
<img src="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/tum_light.svg" width="28%" alt="TUM Logo">
</picture>
<picture>
<img src="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/logo-bavaria.svg" width="16%" alt="Coat of Arms of Bavaria">
</picture>
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/erc_dark.svg" width="24%">
<img src="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/erc_light.svg" width="24%" alt="ERC Logo">
</picture>
<picture>
<img src="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/logo-mqv.svg" width="28%" alt="MQV Logo">
</picture>
</p>
