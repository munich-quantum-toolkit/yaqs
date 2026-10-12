# Saved example outputs

The Markdown files in `docs/examples/` remain the guide sources. This directory
stores their executed notebooks so Read the Docs can publish code, figures, and
results without running simulations or installing PyTorch.

GitHub Actions executes all examples and uploads the notebooks and rendered
pages as the `documentation-examples` artifact. The documentation check waits
for execution and downloads that run's artifact before rendering. Pull requests
do not need local notebook execution or regenerated notebook commits to pass
these GitHub Actions checks. The jobs do not commit changes.

Local execution remains available when you want to inspect new results:

```console
uvx nox --non-interactive -s docs-execute
```

Review the rendered pages in `docs/_build/executed/` or download the CI
artifact. Edit the Markdown sources rather than the saved notebooks. Changes to
prose do not require another execution.

The usual documentation command renders the saved outputs:

```console
uvx nox --non-interactive -s docs
```

Rendering fails if outputs are missing, changed, or stale. The manifest records
the executable cells, notebook settings, package source, dependency inputs, and
execution environment. Generated version strings do not invalidate outputs.
Examples that read additional data files must also include those files in the
input check in `docs/_ext/yaqs_examples.py`.

Read the Docs and ordinary local builds still use the committed snapshots. They
do not download GitHub Actions artifacts. Updating those snapshots for
publication remains a separate step; the CI artifact can supply the new
notebooks and manifest without local execution.
