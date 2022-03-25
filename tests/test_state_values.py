import pytest
from numpy.testing import assert_almost_equal
from dp import StateValueFunction
from dp._types import Policy
from .conftest import SimpleMDP, TState, TAction

gamma = 0.9


@pytest.fixture
def v(test_mdp: SimpleMDP) -> StateValueFunction:
    """Initial state values for use in tests."""
    return StateValueFunction(test_mdp, [3.0, 1.0, 0.0])


@pytest.fixture
def tying_v(test_mdp: SimpleMDP) -> StateValueFunction:
    """Alternative state values with tied optimal actions for state A."""
    return StateValueFunction(test_mdp, [2.0, 20 / 9, 0.0])


@pytest.fixture
def pi() -> Policy[TState, TAction]:
    """Stochastic policy for use in tests."""
    pi_dict = {
        "A": (("L", 0.6), ("R", 0.4)),
        "B": (("L", 1.0),),
        "C": (("L", 1.0),),
    }
    return lambda s: pi_dict[s]


class TestStateValueFunction:
    def test_call_method_returns_value(self, test_mdp: SimpleMDP) -> None:
        v = StateValueFunction(test_mdp)
        for s in test_mdp.states:
            assert v(s) == 0.0

    def test_setting_initial_value_as_scalar(
        self, test_mdp: SimpleMDP
    ) -> None:
        v = StateValueFunction(test_mdp, initial_value=-5.0)
        for s in test_mdp.states:
            assert v(s) == -5.0

    def test_setting_initial_value_as_sequence(
        self, v: StateValueFunction[TState, TAction]
    ) -> None:
        for s, des_v in zip(["A", "B", "C"], [3.0, 1.0, 0.0]):
            assert v(s) == des_v

    def test_to_dict_method(
        self, v: StateValueFunction[TState, TAction]
    ) -> None:
        v_as_dict = v.to_dict()
        assert v_as_dict == {"A": 3.0, "B": 1.0, "C": 0.0}

        # Check this returns a copy rather than the original
        v_as_dict["D"] = 3.14
        assert v.to_dict() != v_as_dict

    @pytest.mark.parametrize(
        "state,action,des_action_value",
        [
            ("A", "L", 0.75 * (1.0 + 0.9 * 0.0) + 0.25 * (-1.0 + 0.9 * 1.0)),
            ("B", "R", 0.75 * (1.0 + 0.9 * 0.0) + 0.25 * (-1.0 + 0.9 * 3.0)),
        ],
    )
    def test_backup_action_value(
        self,
        v: StateValueFunction[TState, TAction],
        state: TState,
        action: TAction,
        des_action_value: float,
    ) -> None:
        act_action_value = v.backup_action_value(state, action, gamma)
        assert_almost_equal(act_action_value, des_action_value)

    @pytest.mark.parametrize(
        "state,expected_v",
        [
            ("A", 0.55 * (1.0 + 0.0) + 0.45 * (-1.0 + 0.9 * 1.0)),
            ("B", 0.75 * (-1.0 + 0.9 * 3.0) + 0.25 * 1.0),
            ("C", 0.0),
        ],
    )
    def test_backup_state_value(
        self,
        v: StateValueFunction[TState, TAction],
        pi: Policy[TState, TAction],
        state: TState,
        expected_v: float,
    ) -> None:
        updated_v = v.backup_state_value(state, pi, gamma)
        assert_almost_equal(updated_v, expected_v)

    @pytest.mark.parametrize(
        "state, expected_action, expected_action_value",
        [
            (TState("A"), TAction("L"), 0.725),
            (TState("B"), TAction("L"), 1.525),
        ],
    )
    def test_backup_optimal_actions(
        self,
        v: StateValueFunction[TState, TAction],
        state: TState,
        expected_action: TAction,
        expected_action_value: float,
    ) -> None:
        actions, action_value = v.backup_optimal_actions(state, gamma)
        assert len(actions) == 1
        assert actions[0] == expected_action
        assert_almost_equal(action_value, expected_action_value)

    def test_backup_optimal_actions_with_tie(
        self,
        tying_v: StateValueFunction[TState, TAction],
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
        actions, action_value = tying_v.backup_optimal_actions(state, gamma)
        assert_almost_equal(action_value, 1.0)
        assert set(actions) == {TAction("L"), TAction("R")}

    def test_optimal_actions_map(
        self, tying_v: StateValueFunction[TState, TAction]
    ) -> None:
        act_map = tying_v.optimal_actions_map(gamma)
        des_map = {"A": {"L", "R"}, "B": {"R"}, "C": {"L", "R"}}
        assert act_map.keys() == des_map.keys()
        for s in act_map.keys():
            assert set(act_map[s]) == des_map[s]
