# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Publish saved notebook outputs and regenerate them in a separate build."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import nbformat
from jupyter_cache import get_cache
from jupyter_cache.base import CacheBundleIn
from myst_nb.core.read import is_myst_markdown_notebook, read_myst_markdown_notebook
from sphinx.errors import SphinxError

if TYPE_CHECKING:
    from nbformat import NotebookNode
    from sphinx.application import Sphinx


def _notebooks(app: Sphinx) -> dict[Path, NotebookNode]:
    """Read notebook sources, excluding generated documentation.

    Returns:
        Source paths relative to the documentation directory and their notebooks.
    """
    source = Path(app.srcdir)
    notebooks = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if relative.parts[0] in {"_build", "_outputs"}:
            continue
        if path.suffix == ".ipynb":
            notebooks[relative] = nbformat.read(path, as_version=4)
        elif path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
            if is_myst_markdown_notebook(text):
                notebooks[relative] = read_myst_markdown_notebook(
                    text, config=app.env.myst_config, add_source_map=True, path=path
                )
    return notebooks


def _runtime_digest(root: Path) -> str:
    """Hash package code and dependency inputs that can change example results.

    Returns:
        A digest independent of paths, timestamps, and generated version strings.
    """
    digest = hashlib.sha256()
    paths = [*sorted((root / "src").rglob("*.py")), root / "pyproject.toml", root / "uv.lock"]
    for path in paths:
        if path.is_file() and path.name != "_version.py":
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _input_digest(notebook: NotebookNode) -> str:
    """Hash executable cells and settings, allowing prose-only edits.

    Returns:
        The digest of inputs used to produce notebook outputs.
    """
    cells = [
        {"source": cell.source, "metadata": {key: value for key, value in cell.metadata.items() if key != "source_map"}}
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]
    metadata = {key: value for key, value in notebook.metadata.items() if key != "source_map"}
    payload = json.dumps({"metadata": metadata, "cells": cells}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _prepare_outputs(app: Sphinx) -> None:
    """Require current saved outputs before publishing, then seed MyST's cache.

    Raises:
        SphinxError: Saved outputs are missing, stale, or cannot be used safely.
    """
    notebooks = _notebooks(app)
    if app.env.mystnb_config.execution_mode != "cache" or any(
        notebook.metadata.get("mystnb", {}).get("execution_mode", "cache") != "cache" for notebook in notebooks.values()
    ):
        msg = "Documentation requires nb_execution_mode='cache' to publish saved outputs safely."
        raise SphinxError(msg)
    if app.config.yaqs_generate_examples:
        return
    output = Path(app.srcdir) / "_outputs"
    instruction = "Regenerate example outputs with: uvx nox --non-interactive -s docs-execute"
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        msg = f"Missing or invalid example output manifest. {instruction}"
        raise SphinxError(msg) from error
    if manifest.get("runtime") != _runtime_digest(Path(app.srcdir).parent):
        msg = f"Example outputs predate changes to package code or dependencies. {instruction}"
        raise SphinxError(msg)
    cache = get_cache(app.env.mystnb_config.execution_cache_path)
    for relative, notebook in notebooks.items():
        entry = manifest.get("notebooks", {}).get(relative.as_posix())
        saved = output / relative.with_suffix(".ipynb")
        if entry is None or entry["inputs"] != _input_digest(notebook):
            msg = f"Missing or stale example outputs for {relative}. {instruction}"
            raise SphinxError(msg)
        if not saved.is_file() or hashlib.sha256(saved.read_bytes()).hexdigest() != entry["sha256"]:
            msg = f"Missing or changed saved notebook {saved}. {instruction}"
            raise SphinxError(msg)
        executed = nbformat.read(saved, as_version=4)
        cache.cache_notebook_bundle(CacheBundleIn(executed, str(Path(app.srcdir) / relative)), overwrite=True)
        try:
            cache.match_cache_notebook(notebook)
        except KeyError as error:
            msg = f"Saved outputs do not match executable cells in {relative}. {instruction}"
            raise SphinxError(msg) from error


def _output_dependencies(app: Sphinx, docname: str, source: list[str]) -> None:
    """Refresh rendered pages when their saved outputs change."""
    del source
    output = Path(app.srcdir) / "_outputs" / f"{docname}.ipynb"
    if output.is_file():
        app.env.note_dependency(str(output))
        app.env.note_dependency(str(Path(app.srcdir) / "_outputs" / "manifest.json"))


def _save_outputs(app: Sphinx, exception: Exception | None) -> None:
    """Export successfully executed notebooks and record their input provenance."""
    if (
        exception is not None
        or app.statuscode
        or not app.config.yaqs_generate_examples
        or app.tags.has("sphinx_llm_markdown")
    ):
        return
    output = Path(app.srcdir) / "_outputs"
    output.mkdir(parents=True, exist_ok=True)
    executed_dir = Path(app.env.mystnb_config.output_folder)
    entries = {}
    for relative, notebook in _notebooks(app).items():
        executed = nbformat.read(executed_dir / relative.with_suffix(".ipynb"), as_version=4)
        for index, cell in enumerate(executed.cells):
            cell.id = f"cell-{index}"
            cell.metadata.pop("execution", None)
        saved = output / relative.with_suffix(".ipynb")
        saved.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(executed, saved)
        entries[relative.as_posix()] = {
            "inputs": _input_digest(notebook),
            "sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
        }
    retained = {output / Path(name).with_suffix(".ipynb") for name in entries}
    for saved in output.rglob("*.ipynb"):
        if saved not in retained:
            saved.unlink()
    manifest = {
        "runtime": _runtime_digest(Path(app.srcdir).parent),
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("mqt.yaqs", "numpy", "scipy", "qiskit", "myst-nb")},
        },
        "notebooks": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def setup(app: Sphinx) -> dict[str, bool]:
    """Register saved-output publishing and generation hooks.

    Returns:
        Parallel build support declarations.
    """
    app.add_config_value("yaqs_generate_examples", os.environ.get("YAQS_DOCS_EXECUTE") == "1", "env")
    app.connect("builder-inited", _prepare_outputs, priority=600)
    app.connect("source-read", _output_dependencies)
    app.connect("build-finished", _save_outputs, priority=400)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
