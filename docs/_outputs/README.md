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

Read the Docs publishes the committed snapshots. It checks executable cells,
notebook settings, and saved-file hashes, but permits a different package or
dependency fingerprint with `yaqs_require_current_runtime=0`. Its figures show
the last committed execution; GitHub Actions validates execution against the
current code. CI and ordinary local builds require a matching runtime by
default.

Refresh publication figures deliberately from the CI artifact when the results
change or before a release. No local execution is needed. Read the Docs does not
download GitHub Actions artifacts or run guide notebooks.
