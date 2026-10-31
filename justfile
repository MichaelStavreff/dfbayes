set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Install the project and its dependencies into .venv
build:
    uv sync

# Run the pytest suite
test:
    uv run pytest -q

# Lint and type-check
lint:
    uvx --from ruff==0.17.3 ruff check
    uvx --from mypy==2.4.1 mypy

# Build sdist and wheel, then validate the metadata
dist:
    uv build
    uvx --from twine==7.0.0 twine check dist/*

# Regenerate golden fixtures by running the R original (requires R + DFBA)
fixtures:
    Rscript tests/fixtures/generate.R

# Also regenerate the slow stochastic fixtures (frozen R intermediates)
fixtures-stochastic:
    DFBA_STOCHASTIC=1 Rscript tests/fixtures/generate.R

# Remove local build and test artefacts
clean:
    rm -rf .venv .pytest_cache build dist src/*.egg-info
    find . -name '__pycache__' -type d -prune -exec rm -rf {} +
