# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Keep API re-exports and type references linked to their defining objects."""

from __future__ import annotations

import builtins
import shutil
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from docutils import nodes
from sphinx import addnodes
from sphinx.domains.python import ObjectEntry, PythonDomain

if TYPE_CHECKING:
    from collections.abc import Iterator

    from autoapi._objects import PythonObject
    from sphinx.application import Sphinx
    from sphinx_markdown_builder.translator import MarkdownTranslator


_EXTERNAL_ALIASES = {
    "NDArray": "numpy.typing.NDArray",
    "ArrayLike": "numpy.typing.ArrayLike",
    "QuantumCircuit": "qiskit.circuit.QuantumCircuit",
    "qiskit.QuantumCircuit": "qiskit.circuit.QuantumCircuit",
    "DAGCircuit": "qiskit.dagcircuit.DAGCircuit",
    "DAGOpNode": "qiskit.dagcircuit.DAGOpNode",
    "Parameter": "qiskit.circuit.Parameter",
    "scipy.linalg.LinAlgError": "numpy.linalg.LinAlgError",
}


class _APIPythonDomain(PythonDomain):
    """Publish public import aliases alongside their canonical API entries."""

    def get_objects(self) -> Iterator[tuple[str, str, str, str, str, int]]:
        """Include import aliases, even when a module has the same name.

        Yields:
            Python inventory entries.
        """
        yield from super().get_objects()
        for name, entry in self.data.get("yaqs_public_aliases", {}).items():
            if name not in self.objects or self.objects[name].objtype != entry.objtype:
                yield name, name, entry.objtype, entry.docname, entry.node_id, -1


def _canonical_name(objects: dict[str, PythonObject], name: str) -> str:
    """Follow a re-export, including a method or attribute below that export.

    Returns:
        The object's name at its defining module.
    """
    visited = set()
    while True:
        parts = name.split(".")
        for end in range(len(parts), 0, -1):
            obj = objects.get(".".join(parts[:end]))
            if obj is not None and obj.imported:
                if end < len(parts) and obj.type not in {"class", "exception", "module", "package"}:
                    continue
                if obj.id in visited:
                    return name
                visited.add(obj.id)
                original = obj.obj["original_path"]
                suffix = ".".join(parts[end:])
                name = original + (f".{suffix}" if suffix else "")
                break
        else:
            return name


def _public_imports(
    app: Sphinx,
    what: str,
    name: str,
    obj: PythonObject,
    skip: bool,  # ruff: ignore[boolean-type-hint-positional-argument] - Sphinx calls event handlers positionally.
    options: list[str],
) -> None:
    """List explicit public re-exports without repeating their documentation."""
    del app, name, options
    if skip or what not in {"module", "package"}:
        return
    exports = obj.obj.get("all") or []
    links = [
        f"* :py:obj:`{child.name} <{child.obj['original_path']}>`"
        for child in obj.children
        if child.imported and child.name in exports
    ]
    if links:
        obj.docstring += "\n\n.. rubric:: Public imports\n\n" + "\n".join(links) + "\n"


def _resolve_api_references(app: Sphinx, doctree: nodes.document) -> None:
    """Qualify local aliases and use inventory roles for external types."""
    objects = getattr(app.env, "autoapi_all_objects", {})
    inventory = getattr(app.env, "intersphinx_inventory", {})
    for node in doctree.findall(addnodes.pending_xref):
        if node.get("refdomain") != "py":
            continue
        target = node["reftarget"]
        if node.get("reftype") in {"class", "obj"} and isinstance(getattr(builtins, target, None), type):
            # Avoid resolving builtin ``type`` or ``float`` to a class member.
            node["reftarget"] = f"python:{target}"
            continue
        module = node.get("py:module", "")
        classname = node.get("py:class", "")
        candidates = [target, f"{module}.{classname}.{target}", f"{module}.{target}"]
        if target.startswith("."):
            candidates.insert(0, module + target)
        # Resolve within the declared scope before searching globally. The
        # imported object remains in AutoAPI's model even when it is not rendered.
        for candidate in candidates:
            if candidate in objects:
                target = _canonical_name(objects, candidate) if node.get("reftype") != "mod" else candidate
                break
        else:
            if "." not in target:
                matches = {
                    name
                    for name, obj in objects.items()
                    if name.endswith(f".{target}") and obj.display and not obj.imported
                }
                if len(matches) == 1:
                    target = matches.pop()

        target = _EXTERNAL_ALIASES.get(target, target)
        if target.startswith("np."):
            target = "numpy." + target[3:]
        obj = objects.get(target)
        if (
            obj is not None and not obj.display and not obj.imported
        ) or target == "multiprocessing.context.BaseContext":
            # Keep known internal types readable without linking to deliberately
            # omitted API entries. Python has no inventory entry for BaseContext.
            # Unknown YAQS names still pass through to Sphinx's strict check.
            node.replace_self(nodes.literal("", node.astext()))
            continue
        node["reftarget"] = target
        if node.get("reftype") == "class" and any(
            target in inventory.get(role, {}) for role in ("py:data", "py:attribute", "py:type")
        ):
            # NumPy documents NDArray as data and scalar dtypes as attributes.
            # Generic object roles retain the exact type and use those entries.
            node["reftype"] = "obj"


def _register_public_aliases(app: Sphinx, env: object) -> None:
    """Keep public re-export names available to downstream intersphinx users."""
    del env
    objects = getattr(app.env, "autoapi_all_objects", {})
    domain = app.env.domains["py"]
    canonical_entries = dict(domain.objects)
    aliases = {}
    for name, obj in objects.items():
        if not obj.imported or obj.obj.get("hide"):
            continue
        canonical = _canonical_name(objects, name)
        for target, entry in canonical_entries.items():
            if target == canonical or target.startswith(f"{canonical}."):
                alias = name + target[len(canonical) :]
                aliases[alias] = ObjectEntry(entry.docname, entry.node_id, entry.objtype, aliased=True)
    domain.data["yaqs_public_aliases"] = aliases


def _markdown_images(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """Keep local image links valid when sphinx-llm merges Markdown outputs."""
    if app.builder.name not in {"llms-markdown", "markdown"}:
        return
    output = Path(app.outdir)
    if app.tags.has("sphinx_llm_markdown"):
        output = output.parent
    for node in doctree.findall(nodes.image):
        image = app.env.images.get(node["uri"])
        if image is None:
            continue
        destination = output / "_images" / image[1]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(app.srcdir) / node["uri"], destination)
        node["uri"] = "../" * docname.count("/") + f"_images/{image[1]}"


def _visit_markdown_abbreviation(translator: MarkdownTranslator, node: nodes.abbreviation) -> None:
    """Keep abbreviation explanations and signature separators in Markdown."""
    translator.add(f'<abbr title="{escape(node.get("explanation", ""), quote=True)}">')


def _depart_markdown_abbreviation(translator: MarkdownTranslator, node: nodes.abbreviation) -> None:
    """Close an abbreviation after the translator has rendered its text."""
    del node
    translator.add("</abbr>")


def setup(app: Sphinx) -> dict[str, bool]:
    """Install API presentation hooks.

    Returns:
        Parallel build support declarations.
    """
    if "autoapi-skip-member" in app.events.events:
        app.connect("autoapi-skip-member", _public_imports)
    app.add_domain(_APIPythonDomain, override=True)
    app.add_node(
        nodes.abbreviation,
        override=True,
        markdown=(_visit_markdown_abbreviation, _depart_markdown_abbreviation),
    )
    app.connect("doctree-read", _resolve_api_references)
    app.connect("env-updated", _register_public_aliases)
    app.connect("doctree-resolved", _markdown_images)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
