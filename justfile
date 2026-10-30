set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Install the project and its dependencies into .venv
build:
    uv sync

# Run the pytest suite
test:
    uv run pytest -q

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
