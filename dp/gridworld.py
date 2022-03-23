from typing import NewType, Tuple, Sequence, Mapping, Optional, Iterable
from itertools import product
from ._types import (
    Policy,
    NextStateProbabilityTable,
    TransitionsMapping,
    Reward,
)
from .base import FiniteMDP


GWState = NewType("GWState", Tuple[int, int])
GWAction = NewType("GWAction", str)
GWPolicy = Policy[GWState, GWAction]
GWNextStateProbabilityTable = NextStateProbabilityTable[GWState]
GWTransitionsMapping = TransitionsMapping[GWState, GWAction]


class GridWorld(FiniteMDP[GWState, GWAction]):
    """A GridWorld finite MDP, generalising the example given in
    Chapter 3 of Sutton, Barto (2018)."""

    def __init__(
        self,
        size: int,
        wormholes: Optional[Mapping[GWState, Tuple[GWState, Reward]]] = None,
        terminal_states: Optional[Iterable[GWState]] = None,
        default_move_reward: float = 0.0,
        invalid_action_reward: float = -1.0,
    ):
        self.size = size
        self.wormholes = wormholes if wormholes is not None else {}
        self.terminal_states = (
            set(terminal_states) if terminal_states is not None else {}
        )
        self._states = [
            GWState((i, j)) for i, j in product(range(size), range(size))
        ]
        self.actions_to_moves = {
            "n": (-1, 0),
            "e": (0, 1),
            "w": (0, -1),
            "s": (1, 0),
        }
        self.default_move_reward = default_move_reward
        self.invalid_action_reward = invalid_action_reward

    @property
    def states(self) -> Sequence[GWState]:
        return self._states

    def actions(self, state: GWState) -> Sequence[GWAction]:
        # This MDP has same actions allowed in any state
        return list(GWAction(a) for a in self.actions_to_moves)

    def s2i(self, state: GWState) -> int:
        return state[0] * self.size + state[1]

    def i2s(self, index: int) -> GWState:
        return GWState((index // self.size, index % self.size))

    def next_states_and_rewards(
        self, state: GWState, action: GWAction
    ) -> Tuple[GWNextStateProbabilityTable, float]:
        if state in self.terminal_states:
            return (
                (state,),
                (1.0,),
            ), 0.0
        elif state in self.wormholes:
            next_state, reward = self.wormholes[state]
            return (
                (next_state,),
                (1.0,),
            ), reward
        else:
            move = self.actions_to_moves[action]
            next_state = GWState((state[0] + move[0], state[1] + move[1]))
            if self.state_is_valid(next_state):
                return (
                    (next_state,),
                    (1.0,),
                ), self.default_move_reward
            else:
                return (
                    (state,),
                    (1.0,),
                ), self.invalid_action_reward

    def state_is_valid(self, state: GWState) -> bool:
        return min(state) >= 0 and max(state) < self.size
