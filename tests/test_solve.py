from typing import Dict
import pytest
from numpy.testing import assert_almost_equal
import numpy as np
from dp._types import Policy
from dp import FiniteMDP
from dp.solve import (
    backup_action_value,
    backup_single_state_value,
    backup_single_state_optimal_actions,
    optimal_actions_from_state_values,
    backup_policy_values_operator,
    backup_optimal_values,
    exact_state_values,
)
from .conftest import TState, TAction


@pytest.fixture
def gamma() -> float:
    return 0.9


@pytest.fixture
def v() -> Dict[TState, float]:
    """Initial state values for use in tests."""
    return {"A": 3.0, "B": 1.0, "C": 0.0}


@pytest.fixture
def tying_v() -> Dict[TState, float]:
    """Alternative state values with tied optimal actions for state A."""
    return {"A": 2.0, "B": 20 / 9, "C": 0.0}


@pytest.fixture
def pi() -> Policy[TState, TAction]:
    """Stochastic policy for use in tests."""
    pi_dict = {
        "A": (("L", 0.6), ("R", 0.4)),
        "B": (("L", 1.0),),
        "C": (("L", 1.0),),
    }
    return lambda s: pi_dict[s]


class TestSolveBasicComponents:
    @pytest.mark.parametrize(
        "state,expected_v",
        [
            ("A", 0.55 * (1.0 + 0.0) + 0.45 * (-1.0 + 0.9 * 1.0)),
            ("B", 0.75 * (-1.0 + 0.9 * 3.0) + 0.25 * 1.0),
            ("C", 0.0),
        ],
    )
    def test_backup_single_state_value(
        self,
        test_mdp: FiniteMDP,
        v: Dict[TState, float],
        pi: Policy[TState, TAction],
        gamma: float,
        state: TState,
        expected_v: float,
    ) -> None:
        updated_v = backup_single_state_value(test_mdp, state, v, gamma, pi)
        assert_almost_equal(updated_v, expected_v)

    @pytest.mark.parametrize(
        "state,action,des_action_value",
        [
            ("A", "L", 0.75 * (1.0 + 0.9 * 0.0) + 0.25 * (-1.0 + 0.9 * 1.0)),
            ("B", "R", 0.75 * (1.0 + 0.9 * 0.0) + 0.25 * (-1.0 + 0.9 * 3.0)),
        ],
    )
    def test_backup_action_value(
        self,
        test_mdp: FiniteMDP,
        v: Dict[TState, float],
        gamma: float,
        state: TState,
        action: TAction,
        des_action_value: float,
    ) -> None:
        act_action_value = backup_action_value(
            test_mdp, state, action, v, gamma
        )
        assert_almost_equal(act_action_value, des_action_value)

    @pytest.mark.parametrize(
        "state, expected_action, expected_action_value",
        [
            (TState("A"), TAction("L"), 0.725),
            (TState("B"), TAction("L"), 1.525),
        ],
    )
    def test_backup_single_state_optimal_actions(
        self,
        test_mdp: FiniteMDP,
        v: Dict[TState, float],
        gamma: float,
        state: TState,
        expected_action: TAction,
        expected_action_value: float,
    ) -> None:
        actions, action_value = backup_single_state_optimal_actions(
            test_mdp, state, v, gamma
        )
        assert len(actions) == 1
        assert actions[0] == expected_action
        assert_almost_equal(action_value, expected_action_value)

    def test_backup_single_state_optimal_actions_with_tie(
        self,
        test_mdp: FiniteMDP,
        tying_v: Dict[TState, float],
        gamma: float,
    ) -> None:
        """Tests the backup_single_state_optimal_actions method in a case
        where there are multiple actions that return the same value."""
        # This time we set up a value function so that, for the chosen
        # values of gamma and the rewards specified in the MDP, both
        # `L` and `R` actions should have equal value from state `A`. As a
        # result, the method should return both actions.

        # With `v[C]==0` and (arbitrarily) `v[A]==2`, we can solve the
        # the simultaneous equations to show that `v[B]` needs to be 20/9
        # for this to be the case, resulting in an action value update for
        # state A of 1.
        state = TState("A")
        actions, action_value = backup_single_state_optimal_actions(
            test_mdp, state, tying_v, gamma
        )
        assert_almost_equal(action_value, 1.0)
        assert set(actions) == {TAction("L"), TAction("R")}

    def test_optimal_actions_from_state_values(
        self, test_mdp: FiniteMDP, tying_v: Dict[TState, float], gamma: float
    ) -> None:
        act_map = optimal_actions_from_state_values(test_mdp, tying_v, gamma)
        des_map = {"A": {"L", "R"}, "B": {"R"}, "C": {"L", "R"}}
        assert act_map.keys() == des_map.keys()
        for s in act_map.keys():
            assert set(act_map[s]) == des_map[s]

    def test_backup_policy_values_operator(
        self,
        test_mdp: FiniteMDP,
        gamma: float,
        pi: Policy[TState, TAction],
    ) -> None:
        A, b = backup_policy_values_operator(test_mdp, gamma, pi)

        # A should be gamma times the transition matrix
        expected_A = gamma * np.array(
            [[0.0, 0.45, 0.55], [0.75, 0.0, 0.25], [0.0, 0.0, 1.0]]
        )
        assert_almost_equal(A, expected_A)
        # b should be a vector of expected reward per starting state
        expected_b = np.array([0.1, -0.5, 0.0])
        assert_almost_equal(b, expected_b)

    def test_backup_optimal_values(
        self, test_mdp: FiniteMDP, v: Dict[TState, float], gamma: float
    ) -> None:
        initial_v_array = np.array(list(v.values()))
        updated_v = backup_optimal_values(test_mdp, initial_v_array, gamma)
        expected_v = [0.725, 1.525, 0.0]
        assert_almost_equal(updated_v, expected_v)


class TestSolvers:
    def test_exact_state_values_regression(
        self, test_mdp: FiniteMDP, pi: Policy[TState, TAction], gamma: float
    ) -> None:
        """Tests output of solver against saved previous run."""
        expected = {"A": -0.14106313, "B": -0.59521761, "C": 0.0}
        actual = exact_state_values(test_mdp, gamma, pi)
        assert actual.keys() == expected.keys()
        for state in actual.keys():
            assert_almost_equal(actual[state], expected[state], err_msg=state)
