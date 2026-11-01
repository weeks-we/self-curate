from pathlib import Path
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture
def tmp_path():
    # Workspace-owned directories inherit ACLs on managed Windows hosts.
    path = ROOT / "runs" / "test-work" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path
