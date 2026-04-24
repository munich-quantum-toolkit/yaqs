#!/usr/bin/env python3
"""Run the entropy-vs-(J, cut) benchmark with random Clifford unitaries.

This is a thin wrapper around ``benchmark_entropy_vs_j_by_cut.py`` that preserves the
same pipeline, outputs, and CLI, but defaults ``--unitary-ensemble clifford``.
"""

from __future__ import annotations

import sys

from benchmark_entropy_vs_j_by_cut import main as _main


def main() -> None:
    argv = sys.argv
    if "--unitary-ensemble" not in argv:
        argv.extend(["--unitary-ensemble", "clifford"])
    _main()


if __name__ == "__main__":
    main()

