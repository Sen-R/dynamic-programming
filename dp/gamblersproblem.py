from typing import List, Tuple
from ._types import NextStateProbabilityTable
from .base import FiniteMDP


class GamblersProblem(FiniteMDP[int, int]):
    """Gambler's problem as described in Ch 4 of Sutton, Barto (2018).

    Args:
      goal: maximum capital, at which episodes terminates, with reward 1
      p_h: probability of heads (resulting in a winning bet)
    """

    def __init__(self, goal: int, p_h: float):
        self.goal = goal
        self.p_h = p_h

    @property
    def states(self) -> List[int]:
        return list(range(0, self.goal + 1))

    def actions(self, state: int) -> List[int]:
        # Note, although a stake of zero is technically always allowed,
        # this will result in the MDP returning to an identical state
        # at the next turn. Hence we disallow this option except when
        # the MDP is already in a terminal state.
        if state == 0 or state == self.goal:
            return [0]
        else:
            return list(range(1, min(state, self.goal - state) + 1))

    def s2i(self, state: int) -> int:
        return state

    def i2s(self, index: int) -> int:
        return index

    def next_states_and_rewards(
        self, state: int, action: int
    ) -> Tuple[NextStateProbabilityTable[int], float]:
        if action == 0:
            return ((state,), (1.0,)), 0.0
        else:
            next_states = (state - action, state + action)
            probabilities = (1.0 - self.p_h, self.p_h)
            exp_reward = self.p_h if (state + action == self.goal) else 0.0
            return (next_states, probabilities), exp_reward
