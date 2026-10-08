# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integration tests for executable documentation builds."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_html_and_markdown_share_notebook_execution(tmp_path: Path) -> None:
    """Both documentation formats retain outputs from one capped notebook run."""
    pytest.importorskip("sphinx")
    pytest.importorskip("myst_nb")
    pytest.importorskip("sphinx_llm.txt")
    pytest.importorskip("pybtex")

    source = tmp_path / "docs"
    source.mkdir()
    configuration = Path(__file__).parents[2] / "docs" / "conf.py"
    (source / "conf.py").write_text(
        configuration.read_text()
        + '\nextensions = ["myst_nb", "sphinx_llm.txt"]\n'
        + 'html_theme = "basic"\n'
        + "html_theme_options = {}\n"
        + "html_static_path = []\n"
        + "html_css_files = []\n"
        + "templates_path = []\n"
    )
    records = tmp_path / "executions.jsonl"
    (source / "index.md").write_text(
        "---\nfile_format: mystnb\nkernelspec:\n  name: python3\nlanguage_info:\n  name: python\n---\n\n"
        "# Executable documentation\n\n"
        "```{code-cell} ipython3\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import numba\n"
        "import numpy as np\n"
        "from threadpoolctl import threadpool_info\n"
        "from mqt.yaqs import Simulator\n\n"
        "np.eye(4) @ np.eye(4)\n"
        "record = {\n"
        '    "thread_limits": {name: os.environ[name] for name in (\n'
        '        "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS",\n'
        '        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",\n'
        "    )},\n"
        '    "numba_threads": numba.get_num_threads(),\n'
        '    "blas_threads": [pool["num_threads"] for pool in threadpool_info()],\n'
        '    "workers": Simulator(show_progress=False).max_workers,\n'
        "}\n"
        f"with Path({str(records)!r}).open('a') as stream:\n"
        "    stream.write(json.dumps(record) + '\\n')\n"
        'print("NOTEBOOK_RESULT_" + str(6 * 7))\n'
        "```\n"
    )
    output = tmp_path / "_build" / "html"
    environment = os.environ.copy()
    for name in (
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "YAQS_MAX_WORKERS",
        "PYTEST_XDIST_WORKER",
    ):
        environment.pop(name, None)
    environment["IPYTHONDIR"] = str(tmp_path / ".ipython")
    environment["JUPYTER_RUNTIME_DIR"] = str(tmp_path / ".jupyter")
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - sys.executable is trusted.
        [sys.executable, "-m", "sphinx", "-W", "-T", "-b", "html", str(source), str(output)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    runs = [json.loads(line) for line in records.read_text().splitlines()]
    assert len(runs) == 1
    assert all(limit == "1" for limit in runs[0]["thread_limits"].values())
    assert runs[0]["numba_threads"] == 1
    assert 1 <= runs[0]["workers"] <= 2
    assert all(threads == 1 for threads in runs[0]["blas_threads"])
    for artifact in ("index.html", "index.html.md", "llms-full.txt"):
        assert "NOTEBOOK_RESULT_42" in (output / artifact).read_text()
