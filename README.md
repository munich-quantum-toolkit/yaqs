[![PyPI](https://img.shields.io/pypi/v/mqt.bench?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.yaqs/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos%20%7C%20windows-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![codecov](https://img.shields.io/codecov/c/github/cda-tum/mqt-bench?style=flat-square&logo=codecov)](https://codecov.io/gh/cda-tum/mqt-yaqs)

![Banner](images/banner.jpeg)

# MQT YAQS: A Tool for Simulating Open Quantum Systems, Noisy Quantum Circuits, and Realistic Quantum Hardware
YAQS (pronounced "yaks" like the animals) is a Python library, primarily focused on simulating open quantum systems, noisy quantum circuits, and designing realistic quantum hardware.

YAQS is part of the [_Munich Quantum Toolkit_ (_MQT_)](https://mqt.readthedocs.io) developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/).

If you have any questions, feel free to create a [discussion](https://github.com/cda-tum/mqt-yaqs/discussions) or an [issue](https://github.com/cda-tum/mqt-yaqs/issues) on [GitHub](https://github.com/cda-tum/mqt-yaqs).

## Features
- **Simulation of Open Quantum Systems**: Simulate large-scale open quantum systems with a paralellized implementation using the Tensor Jump Method (TJM) ([Large-scale stochastic simulation of open quantum systems](https://arxiv.org/abs/2501.17913v1))
- **Equivalence Checking of Quantum Circuits**: Check the equivalence or non-equivalence of quantum circuits with a scalable MPO-based method ([Equivalence Checking of Quantum Circuits via Intermediary Matrix Product Operator](https://arxiv.org/abs/2410.10946))

## Upcoming Features (Check back soon)
- **WIP: Noisy Quantum Circuit Simulation**: Investigate the effect of noise on large quantum circuits
- **WIP: Quantum Hardware Design**: Design better quantum hardware with realistic simulation methods

## Getting Started

1. Clone this repository:
```bash
   git clone https://github.com/aaronleesander/YAQS.git
   cd yaqs
```

2. Create and activate a virtual environment:

On macOS/Linux
```bash
python -m venv venv
source venv/bin/activate
```

On Windows (PowerShell)
```bash
python -m venv venv
.\venv\Scripts\activate.ps1
```

3. Install YAQS in editable mode so that your changes appear immediately:
```bash
pip install -e .
```

4. Check the ```yaqs/examples``` folder for usage details.

## Contributing
Fork the repository and clone your fork.
Create a new branch for your changes.
Commit and push your work, then open a Pull Request.

## Citations
In case you are using MQT YAQS in your work, we would be thankful if you referred to it by citing the relevant publications:

```bibtex
@misc{sander2025_TJM,
      title={Large-scale stochastic simulation of open quantum systems}, 
      author={Aaron Sander and Maximilian Fröhlich and Martin Eigel and Jens Eisert and Patrick Gelß and Michael Hintermüller and Richard M. Milbradt and Robert Wille and Christian B. Mendl},
      year={2025},
      eprint={2501.17913},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2501.17913}, 
}
@misc{sander2024_equivalencechecking,
      title={Equivalence Checking of Quantum Circuits via Intermediary Matrix Product Operator}, 
      author={Aaron Sander and Lukas Burgholzer and Robert Wille},
      year={2024},
      eprint={2410.10946},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2410.10946}, 
}
```

## Acknowledgements
This work received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (grant agreement No. 101001318) and Millenion, grant agreement
No. 101114305). This work was part of the Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus, and has been supported by the BMWK on the basis of a decision by the German Bundestag through project QuaST, as well as by the BMK, BMDW, and the State of Upper Austria in the frame of the COMET program (managed by the FFG).

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