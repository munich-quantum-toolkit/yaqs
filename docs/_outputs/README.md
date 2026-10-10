# Saved example outputs

The Markdown files in `docs/examples/` remain the guide sources. This directory
stores their executed notebooks so Read the Docs can publish code, figures, and
results without running simulations or installing PyTorch.

After changing example code, package code, or dependencies, regenerate the
outputs:

```console
uvx nox --non-interactive -s docs-execute
```

Review the rendered pages in `docs/_build/executed/` and commit the notebooks
and `manifest.json` with the source changes. Edit the Markdown sources rather
than the saved notebooks. Changes to prose do not require another execution.

The usual documentation command renders the saved outputs:

```console
uvx nox --non-interactive -s docs
```

Publishing fails if outputs are missing, changed, or stale. The manifest records
the executable cells, notebook settings, package source, dependency inputs, and
execution environment. Generated version strings do not invalidate outputs.
Examples that read additional data files must also include those files in the
input check in `docs/_ext/yaqs_examples.py`.

GitHub Actions executes all examples in a separate job and uploads the notebooks
and rendered pages as the `documentation-examples` artifact. The job does not
commit changes. Download and review the artifact if you regenerate outputs in
CI. The documentation check validates the committed copies independently.
