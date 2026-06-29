"""Import-level tests for selector APIs."""

from selectors import BaseSelector, CMIMSelector, FACMIMBasicSelector, FACMIMConditionalSelector, FACMIMSubsetAwareSelector


def test_selector_imports() -> None:
    assert BaseSelector is not None
    assert CMIMSelector is not None
    assert FACMIMBasicSelector is not None
    assert FACMIMConditionalSelector is not None
    assert FACMIMSubsetAwareSelector is not None
