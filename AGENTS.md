<!--- This file has been generated from an external template. Please do not modify it directly. -->
<!--- Changes should be contributed to https://github.com/munich-quantum-toolkit/templates. -->

# MQT YAQS

## Python

- Install package: `uv sync`
- Run tests: `uv run pytest`
- Nox test shortcuts: `uvx nox -s tests`, `uvx nox -s minimums`
- Python 3.14 variants: `uvx nox -s tests-3.14`, `uvx nox -s minimums-3.14`

## Documentation

- Sources: `docs/`
- Build docs locally: `uvx nox --non-interactive -s docs`
- Link check: `uvx nox -s docs -- -b linkcheck`

## Tech Stack

### General

- `prek` for pre-commit hooks

### Python

- Python 3.10+
- `uv` for installation, packaging, and tooling
- `ruff` for formatting/linting (configured in `pyproject.toml`)
- `ty` for type checking
- `pytest` for unit tests (located in `test/python/`)
- `nox` for task orchestration (tests, linting, docs)

### Documentation

- `sphinx`
- MyST (Markdown)
- Furo theme

## Development Guidelines

### General

- MUST follow existing code style by checking neighboring files for patterns.
