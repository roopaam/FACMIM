"""Import-level tests for leakage estimators."""

import leakage_estimators


def test_leakage_imports() -> None:
    assert hasattr(leakage_estimators, "estimate_proxy_leakage")
    assert hasattr(leakage_estimators, "estimate_conditional_leakage")
    assert hasattr(leakage_estimators, "leakage_diagnostics")
