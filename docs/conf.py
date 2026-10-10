# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sphinx configuration file."""

from __future__ import annotations

import os
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import pybtex.plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.template import field, href

if TYPE_CHECKING:
    from pybtex.database import Entry
    from pybtex.richtext import HRef
    from sphinx.application import Sphinx

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

# Limit docs kernels and child builds unless the runner supplies a budget.
for name in (
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ.setdefault(name, "1")
# Automatic worker selection reserves one CPU, so this hint permits two workers.
os.environ.setdefault("YAQS_MAX_WORKERS", "3")

# Keep matplotlib/font cache writable and local during docs builds.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "docs" / "_build" / ".mplconfig"))


try:
    version = metadata.version("mqt.yaqs")
except ModuleNotFoundError:
    msg = "mqt.yaqs must be installed to build the documentation"
    raise ModuleNotFoundError(msg) from None

# Filter git details from version
release = version.split("+")[0]

project = "MQT YAQS"
author = "Chair for Design Automation, TUM"
language = "en"
project_copyright = "2025 - 2026, Chair for Design Automation, TUM"

master_doc = "index"

templates_path = ["_templates"]
html_css_files = ["custom.css"]

extensions = [
    "autoapi.extension",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_llm.txt",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinxext.opengraph",
    "yaqs_api",
    "yaqs_examples",
]

source_suffix = [".rst", ".md"]
nitpicky = True

exclude_patterns = [
    "_build",
    "_outputs",
    "**.ipynb_checkpoints",
    "**.jupyter_cache",
    "**jupyter_execute",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

pygments_style = "colorful"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit", None),
    "mqt": ("https://mqt.readthedocs.io/en/stable", None),
    "core": ("https://mqt.readthedocs.io/projects/core/en/stable", None),
    "ddsim": ("https://mqt.readthedocs.io/projects/ddsim/en/stable", None),
    "qmap": ("https://mqt.readthedocs.io/projects/qmap/en/stable", None),
    "qcec": ("https://mqt.readthedocs.io/projects/qcec/en/stable", None),
    "syrec": ("https://mqt.readthedocs.io/projects/syrec/en/stable", None),
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "substitution",
    "deflist",
    "dollarmath",
]
myst_substitutions = {
    "version": version,
}
myst_heading_anchors = 3

# -- Options for {MyST}NB ----------------------------------------------------

nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_cache_path = os.environ.get("YAQS_DOCS_CACHE", str(ROOT / "docs" / "_build" / ".jupyter_cache"))
# MyST-NB does not know sphinx-llm's builder name. Preserve figures instead of
# selecting only their text representation in the generated Markdown.
nb_mime_priority_overrides = [
    ("llms-markdown", mime, priority)
    for priority, mime in enumerate((
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "image/gif",
        "text/markdown",
        "text/latex",
        "text/html",
        "text/plain",
    ))
]

# Reuse HTML doctrees and notebook outputs when generating the Markdown files.
llms_txt_build_parallel = False
llms_txt_full_build = True


class CDAStyle(UnsrtStyle):
    """Custom style for including PDF links."""

    def format_url(self, _e: Entry) -> HRef:  # ruff:ignore[no-self-use]
        """Format URL field as a link to the PDF.

        Returns:
            The formatted URL field.
        """
        url = field("url", raw=True)
        return href()[url, "[PDF]"]


pybtex.plugin.register_plugin("pybtex.style.formatting", "cda_style", CDAStyle)

bibtex_bibfiles = ["lit_header.bib", "refs.bib"]
bibtex_default_style = "cda_style"

copybutton_prompt_text = r"(?:\(\.?venv\) )?(?:\[.*\] )?\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"

modindex_common_prefix = ["mqt.yaqs."]

autoapi_dirs = ["../src/mqt"]
autoapi_python_use_implicit_namespaces = True
autoapi_root = "api"
autoapi_add_toctree_entry = False
autoapi_ignore = [
    "*/**/_version.py",
]
autoapi_options = [
    "members",
    "show-inheritance",
    "special-members",
    "undoc-members",
]
# Do not carry generated pages for removed modules into the next build.
autoapi_keep_files = False
add_module_names = False
toc_object_entries_show_parents = "hide"
python_use_unqualified_type_names = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# AutoAPI already indexes the real attributes. Keep Google-style attribute
# descriptions as fields, without creating a second target for each attribute.
napoleon_use_ivar = True

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "mqt_dark.png",
    "dark_logo": "mqt_light.png",
    "source_repository": "https://github.com/munich-quantum-toolkit/yaqs/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}


def _release_source_links(app: Sphinx, relative_path: Path, parent_docname: str, content: list[str]) -> None:
    """Link repository source files from included release notes to GitHub."""
    del parent_docname
    if relative_path.as_posix() not in {"../CHANGELOG.md", "../UPGRADING.md"}:
        return
    repository = Path(app.srcdir).parent
    for target in re.findall(r"\]\((src/[^)\s]+)\)", content[0]):
        if (repository / target).is_file():
            content[0] = content[0].replace(
                f"]({target})", f"](https://github.com/munich-quantum-toolkit/yaqs/blob/main/{target})"
            )


def setup(app: Sphinx) -> None:
    """Preserve repository links when release notes are included in the docs."""
    app.connect("include-read", _release_source_links)
