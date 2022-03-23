import pytest
from dp._types import Reward
from dp.gridworld import GridWorld, GWState, GWAction


class TestGridWorld:
    @pytest.mark.parametrize(
        "state,action,expected_next_state,expected_reward",
        [
            (GWState((2, 1)), GWAction("e"), GWState((2, 2)), 0),
            (GWState((3, 2)), GWAction("w"), GWState((3, 1)), 0),
            (GWState((2, 4)), GWAction("n"), GWState((1, 4)), 0),
            (GWState((1, 2)), GWAction("s"), GWState((2, 2)), 0),
            (GWState((0, 2)), GWAction("n"), GWState((0, 2)), -1),
            (GWState((4, 1)), GWAction("s"), GWState((4, 1)), -1),
            (GWState((2, 4)), GWAction("e"), GWState((2, 4)), -1),
            (GWState((1, 0)), GWAction("w"), GWState((1, 0)), -1),
            (GWState((0, 1)), GWAction("n"), GWState((4, 1)), 10),
            (GWState((0, 3)), GWAction("e"), GWState((2, 3)), 5),
        ],
    )
    def test_next_states_and_rewards(
        self,
        gridworld: GridWorld,
        state: GWState,
        action: GWAction,
        expected_next_state: GWState,
        expected_reward: Reward,
    ) -> None:
        """Tests whether `next_states_and_rewards` method delivers expected
        next states, rewards and probabilities."""
        assert gridworld.next_states_and_rewards(state, action) == (
            (
                (expected_next_state,),
                (1.0,),
            ),
            expected_reward,
        )

    @pytest.mark.parametrize(
        "state", [GWState((3, 0)), GWState((1, 4)), GWState((0, 0))]
    )
    def test_s2i(self, gridworld: GridWorld, state: GWState) -> None:
        assert gridworld.states[gridworld.s2i(state)] == state

    @pytest.mark.parametrize("idx", [2, 14, 22])
    def test_i2s(self, gridworld: GridWorld, idx: int) -> None:
        assert gridworld.i2s(idx) == gridworld.states[idx]

    def test_terminal_state_functionality(self) -> None:
        gridworld = GridWorld(2, terminal_states=(GWState((1, 0)),))
        (ns, p_ns), r = gridworld.next_states_and_rewards(
            GWState((1, 0)), GWAction("n")
        )
        assert len(ns) == 1
        assert len(p_ns) == 1
        assert ns[0] == GWState((1, 0))
        assert r == 0.0
        assert p_ns[0] == 1.0

    def test_wormholes_set_to_none_means_no_wormholes(self) -> None:
        gridworld = GridWorld(2)
        assert len(gridworld.wormholes) == 0

    def test_terminal_state_set_to_none_means_no_terminal_states(self) -> None:
        gridworld = GridWorld(2)
        assert len(gridworld.terminal_states) == 0

    def test_custom_move_reward(self) -> None:
        gridworld = GridWorld(2, default_move_reward=3.0)
        (ns, p_ns), r = gridworld.next_states_and_rewards(
            GWState((1, 0)), GWAction("e")
        )
        assert len(ns) == 1
        assert len(p_ns) == 1
        assert ns[0] == GWState((1, 1))
        assert r == 3.0
        assert p_ns[0] == 1.0

    def test_custom_invalid_action_reward(self) -> None:
        gridworld = GridWorld(2, invalid_action_reward=5.0)
        (ns, p_ns), r = gridworld.next_states_and_rewards(
            GWState((1, 0)), GWAction("w")
        )
        assert len(ns) == 1
        assert len(p_ns) == 1
        assert ns[0] == GWState((1, 0))
        assert r == 5.0
        assert p_ns[0] == 1.0
