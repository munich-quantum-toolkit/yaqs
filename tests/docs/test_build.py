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
import re
import shlex
import shutil
import subprocess
import sys
import zlib
from pathlib import Path
from string import Template

import pytest


def _write_configuration(source: Path, extensions: list[str], extra: str = "") -> None:
    """Use the real build settings with a small, self-contained documentation tree.

    Args:
        source: Documentation source directory.
        extensions: Extensions needed by this test.
        extra: Additional configuration settings.
    """
    configuration = Path(__file__).parents[2] / "docs" / "conf.py"
    shutil.copytree(configuration.parent / "_ext", source / "_ext")
    (source / "conf.py").write_text(
        configuration.read_text()
        + f"\nextensions = {extensions!r}\n"
        + 'html_theme = "basic"\n'
        + "html_theme_options = {}\n"
        + "html_static_path = []\n"
        + "html_css_files = []\n"
        + "templates_path = []\n"
        + "intersphinx_mapping = {}\n"
        + extra
    )


@pytest.mark.parametrize("missing_source", [False, True])
def test_included_release_links(tmp_path: Path, *, missing_source: bool) -> None:
    """Included release notes link to source and guides without changing the originals."""
    pytest.importorskip("sphinx")
    pytest.importorskip("myst_nb")
    pytest.importorskip("pybtex")
    source = tmp_path / "docs"
    source.mkdir()
    _write_configuration(source, ["myst_nb"], 'nb_execution_mode = "off"\n')
    module = tmp_path / "src" / "example.py"
    module.parent.mkdir()
    if not missing_source:
        module.write_text('"""Example source."""\n')
    releases = {
        "CHANGELOG.md": "# Changelog\n\nSee [example](src/example.py).\n",
        "UPGRADING.md": "# Upgrading\n\nSee [guide](docs/guide.md#time-dependent-hamiltonians).\n",
    }
    for name, text in releases.items():
        (tmp_path / name).write_text(text)
        shutil.copyfile(Path(__file__).parents[2] / "docs" / name, source / name)
    (source / "index.md").write_text("# Release documentation\n\n```{toctree}\nCHANGELOG\nUPGRADING\nguide\n```\n")
    (source / "guide.md").write_text("# Guide\n\n## Time-dependent Hamiltonians\n")
    output = tmp_path / "html"
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - sys.executable is trusted.
        [sys.executable, "-m", "sphinx", "-W", "-T", "-b", "html", str(source), str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    log = completed.stdout + completed.stderr
    for name, text in releases.items():
        assert (tmp_path / name).read_text() == text
    if missing_source:
        assert completed.returncode != 0
        assert "cross-reference target not found: 'src/example.py'" in log
        return
    assert completed.returncode == 0, log
    assert "WARNING:" not in log
    assert (
        'href="https://github.com/munich-quantum-toolkit/yaqs/blob/main/src/example.py"'
        in (output / "CHANGELOG.html").read_text()
    )
    assert 'href="guide.html#time-dependent-hamiltonians"' in (output / "UPGRADING.html").read_text()


def test_html_and_markdown_share_notebook_execution(tmp_path: Path) -> None:
    """Both documentation formats retain outputs from one capped notebook run."""
    pytest.importorskip("sphinx")
    pytest.importorskip("myst_nb")
    pytest.importorskip("sphinx_llm.txt")
    pytest.importorskip("pybtex")
    yaml = pytest.importorskip("yaml")

    source = tmp_path / "docs"
    source.mkdir()
    _write_configuration(source, ["myst_nb", "sphinx_llm.txt", "yaqs_api"])
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
        "from IPython.display import SVG, display\n\n"
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
        'display(SVG(\'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
        '<circle cx="8" cy="8" r="4"/></svg>\'))\n'
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
    environment["READTHEDOCS_VIRTUALENV_PATH"] = sys.prefix
    environment["READTHEDOCS_OUTPUT"] = str(output.parent)
    configuration = yaml.safe_load((Path(__file__).parents[2] / ".readthedocs.yaml").read_text())
    command = configuration["build"]["jobs"]["build"]["html"][0]
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - sys.executable is trusted.
        [Template(argument).substitute(environment) for argument in shlex.split(command)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "WARNING:" not in completed.stdout + completed.stderr
    child_log = re.search(r"Subprocess output available at: (.+)", completed.stdout)
    assert child_log is not None, completed.stdout
    assert "WARNING:" not in Path(child_log[1].strip()).read_text(encoding="utf-8")
    runs = [json.loads(line) for line in records.read_text().splitlines()]
    assert len(runs) == 1
    assert all(limit == "1" for limit in runs[0]["thread_limits"].values())
    assert runs[0]["numba_threads"] == 1
    assert 1 <= runs[0]["workers"] <= 2
    assert all(threads == 1 for threads in runs[0]["blas_threads"])
    for artifact in ("index.html", "index.html.md", "llms-full.txt"):
        assert "NOTEBOOK_RESULT_42" in (output / artifact).read_text()
    markdown = (output / "index.html.md").read_text()
    figures = re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", markdown)
    assert len(figures) == 1, markdown
    assert (output / figures[0]).is_file()
    assert figures[0] in (output / "index.html").read_text()
    assert "image/svg+xml" not in markdown


@pytest.mark.parametrize("missing_reference", [False, True])
def test_api_reexports_and_type_links(tmp_path: Path, *, missing_reference: bool) -> None:
    """Public aliases link to one complete API page, and missing targets still fail."""
    pytest.importorskip("sphinx")
    pytest.importorskip("autoapi")
    pytest.importorskip("sphinx_llm.txt")
    pytest.importorskip("pybtex")
    source = tmp_path / "docs"
    source.mkdir()
    package = tmp_path / "src" / "example_api"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('from .model import Example, model\n__all__ = ["Example", "model"]\n')
    legacy_module = package / "legacy.py"
    legacy_module.write_text('"""A module that will be removed between builds."""\n')
    (package / "model.py").write_text(
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "from numpy.typing import NDArray\n\n"
        "class Example:\n"
        '    """An example with documented attributes.\n\n'
        "    Attributes:\n"
        "        weight (float): Weight used by the example.\n"
        '    """\n\n'
        "    weight: float = 1.0\n\n"
        "    def sample(self, data: NDArray[np.float64], /, *, dtype: type) -> NDArray[np.float64]:\n"
        '        """Return the supplied values.\n\n'
        "        Args:\n"
        "            data (NDArray[np.float64]): Values to return.\n"
        "            dtype (type): Type used for sampling.\n"
        '        """\n'
        "        return data\n\n"
        "def helper() -> int:\n"
        '    """A supported low-level helper."""\n'
        "    return 1\n"
        "\ndef model() -> int:\n"
        '    """A function sharing its module name."""\n'
        "    return 1\n"
    )
    inventory = tmp_path / "types.inv"
    inventory.write_bytes(
        b"# Sphinx inventory version 2\n# Project: External types\n# Version: 1\n"
        b"# The remainder of this file is compressed using zlib.\n"
        + zlib.compress(
            b"numpy.typing.NDArray py:data 1 ndarray.html#numpy.typing.NDArray -\n"
            b"numpy.float64 py:attribute 1 scalar.html#numpy.float64 -\n"
            b"type py:class 1 type.html#type -\n"
            b"float py:class 1 float.html#float -\n"
            b"int py:class 1 int.html#int -\n"
        )
    )
    _write_configuration(
        source,
        ["autoapi.extension", "sphinx.ext.napoleon", "sphinx.ext.intersphinx", "sphinx_llm.txt", "yaqs_api"],
        f"autoapi_dirs = [{str(package)!r}]\n"
        "autoapi_add_toctree_entry = True\n"
        f'intersphinx_mapping = {{"python": ("https://example.invalid/types/", {str(inventory)!r})}}\n',
    )
    (source / "index.rst").write_text(
        "API documentation\n=================\n\n"
        ":class:`example_api.Example` and :meth:`example_api.Example.sample` use "
        ":attr:`example_api.Example.weight`.\n\n"
        "See :func:`example_api.model.helper` for the low-level API.\n\n"
        "A module :mod:`example_api.model` and function :func:`example_api.model` share a name.\n\n"
        ".. toctree::\n\n   api/index\n\n"
        + ("Missing :class:`example_api.DoesNotExist`.\n" if missing_reference else "")
    )
    output = tmp_path / "html"
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - sys.executable is trusted.
        [sys.executable, "-m", "sphinx", "-W", "-T", "-b", "html", str(source), str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    log = completed.stdout + completed.stderr
    if missing_reference:
        assert completed.returncode != 0
        assert "reference target not found: example_api.DoesNotExist" in log
        return
    assert completed.returncode == 0, log
    assert "WARNING:" not in log
    child_log = re.search(r"Subprocess output available at: (.+)", completed.stdout)
    assert child_log is not None, completed.stdout
    assert "WARNING:" not in Path(child_log[1].strip()).read_text(encoding="utf-8")
    public = (output / "api" / "example_api" / "index.html").read_text()
    model = (output / "api" / "example_api" / "model" / "index.html").read_text()
    index = (output / "index.html").read_text()
    assert "Public imports" in public
    assert 'href="model/index.html#example_api.model.Example"' in public
    assert 'id="example_api.Example"' not in public
    assert model.count('id="example_api.model.Example"') == 1
    assert model.count('id="example_api.model.Example.weight"') == 1
    assert "Weight used by the example." in model
    assert 'id="example_api.model.helper"' in model
    description = index.split("<p>", 1)[1].split("</p>", 1)[0]
    for member in ("sample", "weight"):
        assert f'href="api/example_api/model/index.html#example_api.model.Example.{member}"' in description
    collision = index.split("A module", 1)[1].split("</p>", 1)[0]
    assert 'href="api/example_api/model/index.html#module-example_api.model"' in collision
    assert 'href="api/example_api/model/index.html#example_api.model.model"' in collision
    assert 'href="https://example.invalid/types/ndarray.html#numpy.typing.NDArray"' in model
    assert 'href="https://example.invalid/types/scalar.html#numpy.float64"' in model
    assert 'href="https://example.invalid/types/type.html#type"' in model
    markdown = (output / "api" / "example_api" / "model" / "index.html.md").read_text()
    assert '<abbr title="Positional-only parameter separator (PEP 570)">/</abbr>' in markdown
    assert '<abbr title="Keyword-only parameters separator (PEP 3102)">\\*</abbr>' in markdown
    inventory_module = pytest.importorskip("sphinx.util.inventory")
    with (output / "objects.inv").open("rb") as stream:
        published = inventory_module.InventoryFile.load(stream, "", lambda _uri, location: location)
    assert published["py:class"]["example_api.Example"].uri == published["py:class"]["example_api.model.Example"].uri
    assert (
        published["py:method"]["example_api.Example.sample"].uri
        == published["py:method"]["example_api.model.Example.sample"].uri
    )
    assert (
        published["py:attribute"]["example_api.Example.weight"].uri
        == published["py:attribute"]["example_api.model.Example.weight"].uri
    )
    assert "example_api.model" in published["py:module"]
    assert published["py:function"]["example_api.model"].uri.endswith("#example_api.model.model")

    legacy_module.unlink()
    rebuilt = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - sys.executable is trusted.
        [sys.executable, "-m", "sphinx", "-E", "-W", "-T", "-b", "html", str(source), str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert rebuilt.returncode == 0, rebuilt.stdout + rebuilt.stderr
    assert "WARNING:" not in rebuilt.stdout + rebuilt.stderr
    with (output / "objects.inv").open("rb") as stream:
        rebuilt_inventory = inventory_module.InventoryFile.load(stream, "", lambda _uri, location: location)
    assert "example_api.legacy" not in rebuilt_inventory["py:module"]
