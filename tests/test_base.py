import pytest
from dp import FiniteMDP


class TestFiniteMDP:
    @pytest.mark.parametrize("state,index", [("A", 0), ("B", 1), ("C", 2)])
    def test_s2i(self, test_mdp: FiniteMDP, state: str, index: int):
        assert test_mdp.s2i(state) == index

    @pytest.mark.parametrize("state,index", [("A", 0), ("B", 1), ("C", 2)])
    def test_i2s(self, test_mdp: FiniteMDP, state: str, index: int):
        assert test_mdp.i2s(index) == state
