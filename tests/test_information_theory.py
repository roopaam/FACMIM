"""Import-level tests for information theory module."""

import information_theory


def test_information_theory_imports() -> None:
    assert hasattr(information_theory, "mutual_information")
    assert hasattr(information_theory, "conditional_mutual_information")
    assert hasattr(information_theory, "entropy")
