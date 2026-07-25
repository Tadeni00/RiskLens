"""RiskLens Console — Extended Integration Tests (pytest-compatible)"""

import pytest
import requests

BASE = "http://localhost:8000/v1"

TESTS = [
    ("Phase status", "GET", "/phase/bank_ng_gtb"),
    ("Recent scores", "GET", "/recent?limit=5&tenant_id=bank_ng_gtb"),
    ("Drift metrics", "GET", "/drift/bank_ng_gtb"),
    ("Lifecycle", "GET", "/lifecycle/bank_ng_gtb"),
    ("Drift (equity)", "GET", "/drift/bank_ke_equity"),
    ("Lifecycle (yoco)", "GET", "/lifecycle/fintech_za_yoco"),
]


def _is_reachable() -> bool:
    try:
        requests.get(f"{BASE}/phase/bank_ng_gtb", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _is_reachable(), reason="API server not running on localhost:8000"
)
@pytest.mark.parametrize("desc,method,endpoint", TESTS, ids=[t[0] for t in TESTS])
def test_api_extended(desc, method, endpoint):
    r = requests.request(method, f"{BASE}{endpoint}", timeout=5)
    assert r.status_code == 200, f"{desc} returned {r.status_code}: {r.text[:200]}"

    data = r.json()
    assert isinstance(
        data, (dict, list)
    ), f"{desc} returned unexpected type: {type(data)}"

    if isinstance(data, dict):
        assert len(data) > 0, f"{desc} returned empty dict"
    elif isinstance(data, list):
        assert len(data) > 0, f"{desc} returned empty list"
