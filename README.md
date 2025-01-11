![Banner](images/banner.png)

# yaqs (Yet Another Quantum Simulator)
yaqs (Yet Another Quantum Simulator) is a Python library, primarily focused on simulating open quantum systems, noisy quantum circuits, and designing realistic quantum hardware.
This repository follows a “src/” layout to keep the code organized and testable.

## NOTE: This is an ultra-early alpha version released to provide code for various works released in the context of my work (Aaron Sander). A full release will come later once more features are added.

## Features
- **Tensor Jump Method**: Simulate large-scale open quantum systems with a paralellized implementation
- **Equivalence Checking**: Check the equivalence or non-equivalence of quantum circuits with a scalable MPO-based method.
- **WIP: Noisy Quantum Circuit Simulation**: Investigate the effect of noise on large quantum circuits
- **WIP: Quantum Hardware Design**: Design better quantum hardware with realistic simulation methods

## Installation

1. Clone this repository:
   git clone https://github.com/aaronleesander/yaqs.git
   cd yaqs

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On macOS/Linux
(.\venv\Scripts\activate  # On Windows)

3. Install YAQS in editable mode (along with dependencies):
pip install -e .

4. Check the yaqs/examples folder for usage details.

## Contributing
Fork the repository and clone your fork.
Create a new branch for your changes.
Commit and push your work, then open a Pull Request.