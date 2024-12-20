import pytest


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
class TestPSCase:
    """
    This class is used to test specific ps test case
    """
    ...

    def test_something(self, ps_test_case):
        ...
