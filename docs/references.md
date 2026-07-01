```{raw} latex
\begingroup
\renewcommand\section[1]{\endgroup}
\phantomsection
```

````{only} html
# References

*MQT YAQS* implements algorithms from peer-reviewed research.
**When you use YAQS in academic work, please cite the publications that correspond to the features you use:**

- {footcite:p}`sander2025_TJM` for open **analog** system simulation (tensor jump method),
- {footcite:p}`sander2025_CircuitTDVP` for **digital circuit** simulation,
- {footcite:p}`sander2025_EquivalenceChecking` for **equivalence checking**, and
- {footcite:p}`sander2026_computationalregimes` for **trajectory unravellings** and their computational trade-offs.

Representative BibTeX entries:

```bibtex
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
```

YAQS is developed as part of the [Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io).
If you refer to the broader MQT software ecosystem (not a specific YAQS method above), you may additionally cite {footcite:p}`mqt`.

A full bibliography is given below.

```{footbibliography}
:filter: False

sander2025_TJM
sander2025_CircuitTDVP
sander2025_EquivalenceChecking
sander2026_computationalregimes
mqt
```
````
