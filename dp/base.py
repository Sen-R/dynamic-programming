from abc import ABC, abstractmethod
from typing import Generic, Sequence, Tuple
from ._types import State, Action, NextStateProbabilityTable


class FiniteMDP(ABC, Generic[State, Action]):
    """Abstract base class for finite Markov Decision Processes."""

    @property
    @abstractmethod
    def states(self) -> Sequence[State]:
        """Returns list of possible states."""
        pass

    @abstractmethod
    def actions(self, state: State) -> Sequence[Action]:
        """Returns list of all possible actions."""
        pass

    def s2i(self, state) -> int:
        """Converts a state to its corresponding index in `self.states`."""
        return self.states.index(state)

    def i2s(self, index) -> State:
        """Converts a state index to its corresponding state."""
        return self.states[index]

    @abstractmethod
    def next_states_and_rewards(
        self, state: State, action: Action
    ) -> Tuple[NextStateProbabilityTable[State], float]:
        """Returns a probability table for next states, and expected reward,
        after taking `action` in `state`.

        Returns:
          next_states: a tuple of next states and corresponding probabilities
          exp_reward: the expected reward following the chosen action
        """
        pass
