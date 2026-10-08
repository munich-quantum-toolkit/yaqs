# References

MQT YAQS implements algorithms from research publications and preprints. When
you use the library in academic work, please cite both the **YAQS software**
{footcite:p}`YAQS` and the **MQT Handbook** {footcite:p}`mqt`.

We expect to replace the YAQS software citation with a dedicated YAQS
publication. Please use the current software citation for now.

Also cite the research papers for the methods you use:

- {footcite:p}`sander2025_TJM` for open **analog** system simulation (tensor
  jump method),
- {footcite:p}`sander2025_CircuitTDVP` for **digital circuit** simulation,
- {footcite:p}`froehlich2026_NoisyCircuitTJM` for **noisy quantum circuit**
  simulation,
- {footcite:p}`sander2025_EquivalenceChecking` for **equivalence checking**,
- {footcite:p}`sander2026_computationalregimes` for **trajectory unravellings**
  and their computational trade-offs,
- {footcite:p}`ramos2026_NoiseLearning` for **noise characterization**, and
- {footcite:p}`froehlich2026_BUG` for
  **basis-update and Galerkin (BUG) time integration**.

Representative BibTeX entries:

```bibtex
@misc{YAQS,
  author       = {Aaron Sander},
  title        = {{YAQS}: Yet Another Quantum Simulator},
  year         = {2025},
  howpublished = {\url{https://github.com/munich-quantum-toolkit/yaqs}}
}

@article{sander2025_TJM,
  title     = {Large-scale stochastic simulation of open quantum systems},
  author    = {Sander, Aaron and Fr\"{o}hlich, Maximilian and Eigel, Martin and Eisert, Jens and Gel\ss{}, Patrick and Hinterm\"{u}ller, Michael and Milbradt, Richard M. and Wille, Robert and Mendl, Christian B.},
  year      = {2025},
  journal   = {Nature Communications},
  volume    = {16},
  pages     = {11074},
  doi       = {10.1038/s41467-025-66846-x},
}

@misc{sander2025_CircuitTDVP,
  title         = {Quantum circuit simulation with a local time-dependent variational principle},
  author        = {Aaron Sander and Maximilian Fr\"{o}hlich and Mazen Ali and Martin Eigel and Jens Eisert and Michael Hinterm\"{u}ller and Christian B. Mendl and Richard M. Milbradt and Robert Wille},
  year          = {2025},
  eprint        = {2508.10096},
  archiveprefix = {arXiv},
  primaryclass  = {quant-ph},
}

@article{sander2025_EquivalenceChecking,
  title     = {Equivalence checking of quantum circuits via intermediary matrix product operator},
  author    = {Sander, Aaron and Burgholzer, Lukas and Wille, Robert},
  year      = {2025},
  journal   = {Phys. Rev. Res.},
  volume    = {7},
  pages     = {023261},
  doi       = {10.1103/3q71-y8cf},
}

@misc{sander2026_computationalregimes,
  title         = {Computational regimes in matrix-product-state-based quantum trajectory simulations},
  author        = {Aaron Sander and Simon Cichy and Martin Eigel and Jens Eisert and Maximilian Fr\"{o}hlich and Tom Peham and Robert Wille},
  year          = {2026},
  eprint        = {2606.13779},
  archiveprefix = {arXiv},
  primaryclass  = {quant-ph},
}

@misc{froehlich2026_NoisyCircuitTJM,
  title         = {Noisy quantum circuit simulation with the tensor jump method},
  author        = {Maximilian Fr\"{o}hlich and Aaron Sander and Martin Eigel and Robert Wille and Michael Hinterm\"{u}ller},
  year          = {2026},
  url           = {https://arxiv.org/abs/2607.01323},
  eprint        = {2607.01323},
  archiveprefix = {arXiv},
  primaryclass  = {quant-ph},
}

@misc{ramos2026_NoiseLearning,
  title         = {Scalable {Lindblad} Noise Learning via Stochastic Tensor-Network Simulation},
  author        = {Alejandro R. Ramos Ramos and Maximilian Fr\"{o}hlich and Aaron Sander and Robert Wille and Martin Eigel and Patrick Gel\ss{} and Sebastian Pokutta},
  year          = {2026},
  url           = {https://arxiv.org/abs/2608.24668},
  eprint        = {2608.24668},
  archiveprefix = {arXiv},
  primaryclass  = {quant-ph},
}

@misc{froehlich2026_BUG,
  title         = {Basis-update and {Galerkin} time integration in canonical matrix-product-state form},
  author        = {Maximilian Fr\"{o}hlich and Richard M. Milbradt and Martin Eigel and Aaron Sander and Robert Wille and Christian B. Mendl},
  year          = {2026},
  url           = {https://arxiv.org/abs/2608.16994},
  eprint        = {2608.16994},
  archiveprefix = {arXiv},
  primaryclass  = {quant-ph},
}
```

A full list of references is given below.

```{footbibliography}
```
