# MQT YAQS—A Tool for Simulating Open Quantum Systems, Noisy Quantum Circuits, and Realistic Quantum Hardware

```{raw} latex
\begin{abstract}
```

YAQS (pronounced "yaks" like the animals) is a Python library, primarily focused on simulating open quantum systems, noisy quantum circuits, and designing realistic quantum hardware.

YAQS is part of the _{doc}`Munich Quantum Toolkit (MQT) <mqt:index>`_ developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/).

This documentation provides a comprehensive guide to the MQT YAQS library, including {doc}`installation instructions <installation>`, notebook-like examples, and detailed {doc}`API documentation <api/mqt/yaqs/index>`.
The source code of MQT YAQS is publicly available on GitHub at [cda-tum/mqt-yaqs](https://github.com/cda-tum/mqt-yaqs), while pre-built binaries are available via [PyPI](https://pypi.org/project/mqt.yaqs/) for all major operating systems and all modern Python versions.

````{only} latex
```{note}
A live version of this document is available at [mqt.readthedocs.io/projects/yaqs](https://mqt.readthedocs.io/projects/yaqs).
```
````

```{raw} latex
\end{abstract}

\sphinxtableofcontents
```

```{toctree}
:hidden:

self
```

```{toctree}
:maxdepth: 2
:caption: User Guide
installation
examples/physics_simulation
examples/strong_circuit_simulation
examples/weak_circuit_simulation
examples/equivalence_checking
references
```

````{only} not latex
```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Developers
:glob:

contributing
support
development_guide
```
````

```{toctree}
:hidden:
:caption: API Reference

api/mqt/yaqs/index
```
