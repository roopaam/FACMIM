"""Import-level tests for Pareto utilities."""

import pareto


def test_pareto_imports() -> None:
    assert hasattr(pareto, "pareto_front")
