# MQT YAQS—A Tool for Simulating Open Quantum Systems, Noisy Quantum Circuits, and Realistic Quantum Hardware

```{raw} latex
\begin{abstract}
```

YAQS (pronounced "yaks" like the animals) is a Python library, primarily focused on simulating open quantum systems, noisy quantum circuits, and designing realistic quantum hardware.
It is developed as part of the [Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io) by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de).

This documentation provides a comprehensive guide to the MQT YAQS library, including {doc}`installation instructions <installation>`, notebook-like examples, and detailed {doc}`API documentation <api/mqt/yaqs/index>`.
The source code of MQT YAQS is publicly available on GitHub at [munich-quantum-toolkit/yaqs](https://github.com/munich-quantum-toolkit/yaqs), while pre-built binaries are available via [PyPI](https://pypi.org/project/mqt.yaqs/) for all major operating systems and all modern Python versions.

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
examples/analog_simulation
examples/strong_circuit_simulation
examples/weak_circuit_simulation
examples/equivalence_checking
examples/fermi_hubbard_simulation
references
CHANGELOG
UPGRADING
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

```{only} html
## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<div style="margin-top: 0.5em">
<div class="only-light" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Banner">
</div>
<div class="only-dark" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%" alt="MQT Banner">
</div>
</div>

Thank you to all the contributors who have helped make MQT YAQS a reality!

<p align="center">
<a href="https://github.com/munich-quantum-toolkit/yaqs/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/yaqs" />
</a>
</p>
```
