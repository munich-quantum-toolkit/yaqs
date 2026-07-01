```{raw} latex
\begingroup
\renewcommand\section[1]{\endgroup}
\phantomsection
```

````{only} html
# References

*MQT YAQS* has a strong foundation in peer‑reviewed research.
Many of its built‑in algorithms are based on methods published in scientific journals and conferences.

*MQT YAQS* is part of the Munich Quantum Toolkit, which is described in {footcite:p}`mqt`.
If you want to cite the Munich Quantum Toolkit, please use the following BibTeX entry:

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

If you use *MQT YAQS* in your work, we would appreciate if you cited

- {footcite:p}`sander2025_TJM` for simulating open analog quantum systems,
- {footcite:p}`sander2025_CircuitTDVP` for quantum circuit (digital) simulation,
- {footcite:p}`sander2025_EquivalenceChecking` for the equivalence checking algorithm, and
- {footcite:p}`sander2026_computationalregimes` for information about selecting unravellings and their computational implications.

A full list of references is given below.
````

```{footbibliography}
:filter: False

mqt
sander2025_TJM
sander2025_CircuitTDVP
sander2025_EquivalenceChecking
sander2026_computationalregimes
```
