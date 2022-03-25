from typing import Generic, List, Tuple, Dict, Union, Sequence
from collections.abc import Iterable
from numpy.typing import NDArray
from ._types import State, Action, Policy
import numpy as np
from .base import FiniteMDP


class StateValueFunction(Generic[State, Action]):
    """State value function for an MDP.

    Has numerical methods for estimating state values for a given MDP.

    Args:
      mdp: the FiniteMDP to which this state value function is associated
      initial_value: initial value of state value function, can be a scalar
        or a numpy array with length equal to the number of states
    """

    def __init__(
        self,
        mdp: FiniteMDP[State, Action],
        initial_value: Union[float, Sequence[float], NDArray] = 0.0,
    ):
        self.mdp = mdp
        if isinstance(initial_value, float):
            self._v = {s: initial_value for s in mdp.states}
        elif isinstance(initial_value, Iterable):
            self._v = {s: float(v) for s, v in zip(mdp.states, initial_value)}
        else:
            raise RuntimeError("Invalid argument for `initial_value`")

    def to_dict(self) -> Dict[State, float]:
        """Returns dictionary representation of this state value function."""
        return self._v.copy()

    def __call__(self, state: State) -> float:
        return self._v[state]

    def backup_action_value(
        self, state: State, action: Action, gamma: float
    ) -> float:
        """Estimates action value for given state and action from successor
        state values.

        Args:
          state: state for which action value is being estimated
          action: action (in `state`) for which value is being estimated
          gamma: discount factor

        Returns:
          estimated action value
        """
        (nss, ps), exp_r = self.mdp.next_states_and_rewards(state, action)
        action_value = exp_r + gamma * sum(
            p * self._v[ns] for ns, p in zip(nss, ps)
        )
        return action_value

    def backup_state_value(
        self, state: State, pi: Policy[State, Action], gamma: float
    ) -> float:
        """Updates estimated value of `state` from estimated value of
        successor states under policy `pi`.

        Args:
          state: state whose value is to be updated
          pi: policy encoded as conditional probabilities of actions given
          states
          gamma: discount factor

        Returns:
          updated value estimate for `state`
        """
        state_value = sum(
            p_a * self.backup_action_value(state, a, gamma)
            for a, p_a in pi(state)
        )
        return state_value

    def backup_optimal_actions(
        self, state: State, gamma: float
    ) -> Tuple[List[Action], float]:
        """Returns the actions and corresponding value that maximise expected
        return from `state`, as estimated using this state value function

        Args:
          state: current state, for which optimal action is estimated
          gamma: discount factor

        Returns:
          actions: actions that maximise the action value (could be more than
            one)
          action_value: corresponding maximising action value
        """
        available_actions = self.mdp.actions(state)
        all_action_values = [
            self.backup_action_value(state, action, gamma)
            for action in available_actions
        ]
        action_value = max(all_action_values)
        is_maxing = np.isclose(all_action_values, action_value)
        actions = [a for a, m in zip(available_actions, is_maxing) if m]
        assert len(actions) > 0
        return actions, action_value

    def optimal_actions_map(self, gamma: float) -> Dict[State, List[Action]]:
        """Returns dict mapping states to the optimal actions available
        for each state according to this current state value function."""
        return {
            s: self.backup_optimal_actions(s, gamma)[0]
            for s in self.mdp.states
        }
