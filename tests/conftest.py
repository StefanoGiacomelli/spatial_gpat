"""pytest configuration: repository root on ``sys.path``, no bytecode, data guards."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def _data_present(*parts: str) -> bool:
    return (ROOT.joinpath("data", *parts)).exists()


def requires_data(*parts: str):
    """Skip a test when ``data/<parts>`` is absent (reason names the missing path)."""
    return pytest.mark.skipif(
        not _data_present(*parts),
        reason=f"dataset not available: data/{'/'.join(parts)}",
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT
