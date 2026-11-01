"""Package-level metadata checks.

The release workflow refuses to publish unless the tag, ``pyproject.toml`` and
``dfbayes.__version__`` agree. These tests catch the same drift in CI.
"""

from __future__ import annotations

import pathlib
from importlib.metadata import version

import dfbayes


def test_version_matches_installed_metadata() -> None:
    assert dfbayes.__version__ == version("dfbayes")


def test_py_typed_marker_is_shipped() -> None:
    package_dir = pathlib.Path(dfbayes.__file__).parent
    assert (package_dir / "py.typed").is_file()
