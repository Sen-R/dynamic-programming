from typing import Iterable, Dict, List
import pytest
from numpy.testing import assert_almost_equal
from dp._types import NextStateProbabilityTable
from dp import FiniteMDP
from dp.gamblersproblem import GamblersProblem
from dp.solve import value_iteration, optimal_actions_from_state_values


@pytest.fixture
def gp() -> FiniteMDP:
    return GamblersProblem(goal=100, p_h=0.4)


class TestGamblersProblem:
    def test_states_property(self, gp: FiniteMDP) -> None:
        expected = list(range(0, 101))
        actual = list(gp.states)
        assert actual == expected

    @pytest.mark.parametrize(
        "state,expected_actions",
        [
            (3, range(1, 4)),
            (90, range(1, 11)),
            (0, [0]),
            (100, [0]),
        ],
    )
    def test_actions_method(
        self, gp: FiniteMDP, state: int, expected_actions: Iterable[int]
    ) -> None:
        actual_actions = gp.actions(state)
        assert actual_actions == list(expected_actions)

    @pytest.mark.parametrize("state,index", [(0, 0), (13, 13), (100, 100)])
    def test_s2i_i2s(self, gp: FiniteMDP, state: int, index: int) -> None:
        assert gp.s2i(state) == index
        assert gp.i2s(index) == state

    @pytest.mark.parametrize(
        "state,action,des_ns_ptable,des_exp_r",
        [
            (25, 10, ((15, 35), (0.6, 0.4)), 0.0),
            (0, 0, ((0,), (1.0,)), 0.0),
            (13, 0, ((13,), (1.0,)), 0.0),
            (75, 25, ((50, 100), (0.6, 0.4)), 0.4),
            (100, 0, ((100,), (1.0,)), 0.0),
        ],
    )
    def test_next_states_and_rewards(
        self,
        gp: FiniteMDP,
        state: int,
        action: int,
        des_ns_ptable: NextStateProbabilityTable[int],
        des_exp_r: float,
    ) -> None:
        act_ns_ptable, act_exp_r = gp.next_states_and_rewards(state, action)
        assert_almost_equal(act_exp_r, des_exp_r)
        assert act_ns_ptable[0] == des_ns_ptable[0]
        assert_almost_equal(act_ns_ptable[1], des_ns_ptable[1])


@pytest.fixture
def optimal_v(gp: FiniteMDP) -> Dict[int, float]:
    """Returns optimal state values for problem, using value iteration."""
    v = {s: 0.0 for s in gp.states}
    value_iteration(v, gp, 1.0, tol=1e-8)
    return v


@pytest.fixture
def optimal_actions_map(
    gp: FiniteMDP, optimal_v: Dict[int, float]
) -> Dict[int, List[int]]:
    return optimal_actions_from_state_values(gp, optimal_v, 1.0)


# Some known optimal policies for Gambler's problem
def opt_pol_1(state) -> int:
    """'100 or bust' policy.

    Bet the maximum stake required to (possibly) get to 100 in one go.
    """
    return min(state, 100 - state)


def opt_pol_2(state) -> int:
    """'50 or 100 or bust' policy.

    When capital is less than 50, implement a '50 or bust' policy; when
    greater than 50, implement '100 or 50' policy; when exactly 50, go
    all in.
    """
    if state < 50:
        return min(state, 50 - state)
    elif state == 50:
        return 50
    else:
        return min(state - 50, 100 - state)


def opt_pol_3(state) -> int:
    """The optimal policy illustrated in the textbook.

    When capital is 50, stake 50; when capital is 25 or 75 stake 25 (to
    get to either 50 or 100); otherwise stake the amount required to get
    to the nearest multiple of 25.
    """
    if state < 25:
        return min(state, 25 - state)
    elif state == 25:
        return 25
    elif state > 25 and state < 50:
        return min(state - 25, 50 - state)
    elif state == 50:
        return 50
    elif state > 50 and state < 75:
        return min(state - 50, 75 - state)
    elif state == 75:
        return 25
    else:
        return min(state - 75, 100 - state)


class TestGamblersProblemSolution:
    @pytest.mark.parametrize(
        "test_policy",
        [
            {s: opt_pol_1(s) for s in range(101)},
            {s: opt_pol_2(s) for s in range(101)},
            {s: opt_pol_3(s) for s in range(101)},
        ],
    )
    def test_optimal_policies_are_as_expected(
        self,
        optimal_actions_map: Dict[int, List[int]],
        test_policy: Dict[int, int],
    ) -> None:
        """Each of these test policies should be an optimal policy."""
        assert test_policy.keys() == optimal_actions_map.keys()
        for s in test_policy.keys():
            assert test_policy[s] in optimal_actions_map[s]
