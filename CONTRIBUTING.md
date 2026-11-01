# Contributing to `dfbayes`

`dfbayes` is a port of the R package DFBA 0.1.0. **The port is correct when it
reproduces the R original, not when it looks idiomatic on its own.** If the port
and R disagree, the port is wrong.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and a `justfile`:

```sh
uv sync
just test
```

`just` is optional; the recipes are thin wrappers:

| Recipe | Runs |
| --- | --- |
| `just build` | `uv sync` |
| `just test` | `uv run pytest -q` |
| `just lint` | `ruff check` and `mypy` |
| `just dist` | `uv build` and `twine check dist/*` |
| `just fixtures` | regenerate the deterministic fixtures (needs R) |
| `just fixtures-stochastic` | also regenerate the stochastic fixtures (slow) |

CI runs the suite on Python 3.10–3.14, on a minimal-dependency install, and on
the declared dependency floor (numpy 1.24.4 / scipy 1.10.1). It also runs ruff,
mypy, and a build job that installs both the wheel and the sdist.

## Fixtures are the oracle

Expected values come from running R's DFBA 0.1.0, never from hand-written
constants and never from another port.

```sh
just fixtures             # deterministic cases
just fixtures-stochastic  # frozen R intermediates for the stochastic paths
```

Regenerating needs R 4.6.1 and DFBA 0.1.0; the upstream sources are vendored
read-only under `upstream/` and must not be edited. CI does not install R — it
replays the committed fixtures.

Do not loosen a numeric tolerance to make a test pass. Derive the tolerance from
the fixtures and record the rationale in `DECISION.md`.

## Documentation to update with a change

- `DECISION.md` — scope, the RNG policy, and every deliberate divergence from R,
  with its rationale.
- `COVERAGE.md` — the map from each R source branch to the test that pins it.

## Releasing

Publishing uses PyPI Trusted Publishing, so no API token is stored in the
repository. `.github/workflows/publish.yml` runs the suite, verifies the version,
builds and checks the distributions, installs the wheel and sdist in clean
environments, and only then uploads.

### One-time setup

Register a trusted publisher for `dfbayes` on both indices you intend to use.
They are separate services with separate accounts and separate publisher lists.

| | PyPI (real releases) | TestPyPI (optional dry run) |
| --- | --- | --- |
| Publisher page | <https://pypi.org/manage/account/publishing/> | <https://test.pypi.org/manage/account/publishing/> |
| Project | `dfbayes` | `dfbayes` |
| Owner | `MichaelStavreff` | `MichaelStavreff` |
| Repository | `dfbayes` | `dfbayes` |
| Workflow name | `publish.yml` | `publish.yml` |
| Environment | `pypi` | `testpypi` |

The workflow name is the file name (`publish.yml`), not the workflow's `name:`.
A publisher registered on `pypi.org` has no effect on `test.pypi.org`.

### Cutting a release

1. Bump the version in **both** `pyproject.toml` and `src/dfbayes/__init__.py`.
2. Add `.github/release-notes/<tag>.md` with the release notes. Without it the
   workflow falls back to GitHub's generated notes, which are built from merged
   pull requests.
3. Commit, let CI go green on `main`, then tag and push:

   ```sh
   git tag -a v0.1.1 -m "v0.1.1"
   git push origin v0.1.1
   ```

The tag push publishes to PyPI and creates the GitHub release with the wheel and
sdist attached. The workflow refuses to publish if the tag, `pyproject.toml` and
`dfbayes.__version__` disagree.

Do **not** create the GitHub release by hand first: the `release` job would fail
because the release already exists. Do **not** move or delete a tag that has
already published — PyPI rejects re-uploading a version.

### Manual dispatch

`publish.yml` can be dispatched from the Actions tab. It publishes to TestPyPI
by default; selecting `pypi` publishes the current commit for real, which then
cannot be published again by a tag for the same version.

If a dispatch to `pypi` is intended as a dry run, use TestPyPI instead. Its
step allows re-runs of an already-uploaded version (`skip-existing`), which the
PyPI step never does.
