![Banner](images/banner.png)
## NOTE: This is an early alpha version released to provide code for various works released in the context of my work (Aaron Sander). A full release will come later once more features are added.

# YAQS
YAQS (pronounced like the animals "yaks") is a Python library, primarily focused on simulating open quantum systems, noisy quantum circuits, and designing realistic quantum hardware.
This repository follows a “src/” layout to keep the code organized and testable.

## Features
- **Simulation of Open Quantum Systems**: Simulate large-scale open quantum systems with a paralellized implementation using the Tensor Jump Method (TJM)
- **Equivalence Checking of Quantum Circuits**: Check the equivalence or non-equivalence of quantum circuits with a scalable MPO-based method
- **WIP: Noisy Quantum Circuit Simulation**: Investigate the effect of noise on large quantum circuits
- **WIP: Quantum Hardware Design**: Design better quantum hardware with realistic simulation methods

## Installation

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