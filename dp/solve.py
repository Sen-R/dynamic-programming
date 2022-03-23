from typing import Dict, Optional, MutableMapping, Mapping, Tuple, List
from warnings import warn
from tqdm import tqdm  # type: ignore
from numpy.typing import NDArray
import numpy as np
from scipy import optimize  # type: ignore
from ._types import State, Action, Policy
from .base import FiniteMDP


def backup_action_value(
    mdp: FiniteMDP[State, Action],
    state: State,
    action: Action,
    v: Mapping[State, float],
    gamma: float,
) -> float:
    """Estimates action value for given state and action from successor state
    values.

    Args:
      mdp: FiniteMDP for which action value is being estimated
      state: state for which action value is being estimated
      action: action (in `state`) for which value is being estimated
      v: current state value estimates
      gamma: discount factor"""
    (nss, ps), exp_r = mdp.next_states_and_rewards(state, action)
    action_value = exp_r + gamma * sum(p * v[ns] for ns, p in zip(nss, ps))
    return action_value


def backup_single_state_value(
    mdp: FiniteMDP[State, Action],
    state: State,
    v: Mapping[State, float],
    gamma: float,
    pi: Policy[State, Action],
) -> float:
    """Updates estimated value of `state` from estimated value of
    successor states under policy `pi`.

    Args:
      mdp: FiniteMDP whose state value function is being estimated
      state: state whose value is to be updated
      v: current mapping from states to values that is to be updated
      gamma: discount factor
      pi: policy encoded as conditional probabilities of actions given
        states

    Returns:
      updated value estimate for `state`
    """
    state_value = sum(
        p_a * backup_action_value(mdp, state, a, v, gamma)
        for a, p_a in pi(state)
    )
    return state_value


def backup_single_state_optimal_actions(
    mdp: FiniteMDP[State, Action],
    state: State,
    v: Mapping[State, float],
    gamma: float,
) -> Tuple[List[Action], float]:
    """Returns the actions and corresponding value that maximise expected
    return from `state`, estimated using the current state value mapping
    `v`.

    Args:
      mdp: FiniteMDP for which optimal actions are being estimated
      state: current state, for which optimal action is estimated
      v: estimated state values, used to back-up optimal action
      gamma: discount factor

    Returns:
      actions: actions that maximise the action value (could be more than
        one)
      action_value: corresponding maximising action value
    """
    available_actions = mdp.actions(state)
    all_action_values = [
        backup_action_value(mdp, state, action, v, gamma)
        for action in available_actions
    ]
    action_value = max(all_action_values)
    is_maxing = np.isclose(all_action_values, action_value)
    actions = [a for a, m in zip(available_actions, is_maxing) if m]
    assert len(actions) > 0
    return actions, action_value


def optimal_actions_from_state_values(
    mdp: FiniteMDP[State, Action],
    v: Mapping[State, float],
    gamma: float,
) -> Dict[State, List[Action]]:
    return {
        s: backup_single_state_optimal_actions(mdp, s, v, gamma)[0]
        for s in mdp.states
    }


def backup_policy_values_operator(
    mdp: FiniteMDP[State, Action],
    gamma: float,
    pi: Policy[State, Action],
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Returns the matrix and vector components of the Bellman policy
    evaluation operator for this MDP.

    Args:
      mdp: FiniteMDP for which policy values are being estimated
      gamma: discount factor
      pi: function that returns the probability of the given action being
        taken in the given state, according to the agent's policy

    Returns:
      A: matrix component of the Bellman operator, `gamma *t(s, s')`
        where `t` is the state transition matrix under the given policy
      b: vector component of the Bellman operator, i.e. the expected
        reward given state `s` (marginalising over all possible actions
        and transitioned states)
    """
    expected_rewards_vector = np.zeros(len(mdp.states))
    discounted_transitions_matrix = np.zeros(
        (len(mdp.states), len(mdp.states))
    )

    for s in mdp.states:
        for a, p_a in pi(s):
            ns_ptable, r = mdp.next_states_and_rewards(s, a)
            expected_rewards_vector[mdp.s2i(s)] += p_a * r
            for ns, p_ns in zip(*ns_ptable):
                discounted_transitions_matrix[mdp.s2i(s), mdp.s2i(ns)] += (
                    gamma * p_a * p_ns
                )

    return discounted_transitions_matrix, expected_rewards_vector


def backup_optimal_values(
    mdp: FiniteMDP[State, Action],
    initial_values: NDArray[np.float_],
    gamma: float,
) -> NDArray[np.float_]:
    """Single update of the state-value function; RHS of Bellman
    optimality equation.

    Args:
      mdp: FiniteMDP for which optimal state values are being estimated
      initial_values: array of initial state values to back-up
      gamma: discount factor

    Returns:
      array of updated values
    """
    initial_values = np.array(initial_values)
    updated_values = np.zeros(len(mdp.states))
    for s in mdp.states:
        greatest_action_value = -np.inf
        for a in mdp.actions(s):
            ns_ptable, r = mdp.next_states_and_rewards(s, a)
            greatest_action_value = max(
                greatest_action_value,
                sum(
                    p_ns * (r + gamma * initial_values[mdp.s2i(ns)])
                    for ns, p_ns in zip(*ns_ptable)
                ),
            )
        updated_values[mdp.s2i(s)] = greatest_action_value
    return updated_values


def exact_state_values(
    mdp: FiniteMDP[State, Action],
    gamma: float,
    pi: Policy[State, Action],
) -> Dict[State, float]:
    """Returns state values for given policy.

    This function directly solves the (linear) Bellman equation to calculate
    the state value function for the policy represented by `pi`.

    Args:
      mdp: FiniteMDP for which state values are being calculated
      gamma: discount factor
      pi: conditional probabilities for actions given states, encoding the
        policy to be evaluated

    Returns:
      `dict` mapping states to state values
    """
    A, b = backup_policy_values_operator(mdp, gamma, pi)
    v = np.linalg.solve(np.eye(len(mdp.states)) - A, b)
    return {mdp.i2s(idx): value for idx, value in enumerate(v)}


def exact_optimum_state_values(
    mdp: FiniteMDP[State, Action], gamma: float, tol: Optional[float] = None
) -> Dict[State, float]:
    """Returns state values for an optimal policy for the given MDP.

    This function uses a non-linear solver to directly solve the Bellman
    optimality equation, to calculate state values for an optimal policy.

    Args:
      mdp: FiniteMDP for which optimal state values are being calculated
      gamma: discount factor

    Returns:
      `dict` mapping states to state values under an optimal policy
    """
    initial_guess = np.zeros(len(mdp.states))
    opt_result = optimize.root(
        lambda v_star: v_star - backup_optimal_values(mdp, v_star, gamma),
        x0=initial_guess,
        tol=tol,
    )

    if not opt_result.success:
        raise optimize.OptimizeWarning(
            "Root finding failed to find a solution", opt_result
        )

    return {mdp.i2s(idx): val for idx, val in enumerate(opt_result.x)}


def iterative_policy_evaluation(
    v: MutableMapping[State, float],
    mdp: FiniteMDP[State, Action],
    gamma: float,
    pi: Policy[State, Action],
    tol: Optional[float],
    maxiter: int = 100,
) -> int:
    """Applies iterative policy evaluation to refine provided state value
    estimates.

    Args:
      v: initial estimates of state values which are refined in place
      mdp: FiniteMDP for which state values are being estimated
      gamma: discount factor
      pi: conditional probabilities for actions given states, encoding the
        policy being evaluated
      tol: iteration terminates when maximum absolute change in state value
        function falls below this value, alternative set to `None` and
        iteration will proceed until maxiter is reached
      maxiter: iteration terminates when the number of sweeps through the
        MDP's state space reaches this value

    Returns:
      niter: number of sweeps of the state space that were completed
    """
    for niter in range(1, maxiter + 1):
        delta_v = 0.0  # tracks biggest change to v so far
        for s in mdp.states:
            v_old = v[s]
            v[s] = backup_single_state_value(mdp, s, v, gamma, pi)
            delta_v = max(delta_v, abs(v[s] - v_old))
        if tol is not None and delta_v < tol:
            break
    else:
        # Loop completed normally implying maxiter was reached. If tol is
        # not None, these means the solution has not converged to the tolerance
        # expected.

        if tol is not None:
            warn(
                "`maxiter` sweeps were completed before solution converged to "
                "within desired tolerance, try increasing either `maxiter` or "
                "`tol`"
            )
    return niter


def policy_iteration(
    v: MutableMapping[State, float],
    pi: MutableMapping[State, Action],
    mdp: FiniteMDP[State, Action],
    gamma: float,
    tol: float,
    maxiter: int = 100,
) -> int:
    """Performs policy iteration to refine policy `pi` and corresponding
    state value estimates `v`.

    Args:
      v: mapping from states to state values for current policy
      pi: initial (deterministic) policy, mapping states to actions
      mdp: FiniteMDP for which optimal policy is being searched
      gamma: discount factor
      tol: tolerance for embedded policy evaluation component of each policy
        iteration to converge
      maxiter: maximum number of policy iterations to perform

    Returns:
      niter: number of sweeps of the state space that were performed
    """
    for niter in tqdm(range(1, maxiter + 1)):
        policy_stable = True
        for s in mdp.states:
            old_a = pi[s]
            new_as, _ = backup_single_state_optimal_actions(mdp, s, v, gamma)
            if old_a not in new_as:
                policy_stable = False
                pi[s] = new_as[0]  # Arbitrarily pick one if there are many
        if policy_stable:
            break
        iterative_policy_evaluation(
            v, mdp, gamma, (lambda s: ((pi[s], 1.0),)), tol
        )
    else:
        # maxiter reached but policy not yet stable, so warn
        warn("maxiter reached but policy not yet stable")
    return niter


def value_iteration(
    v: MutableMapping[State, float],
    mdp: FiniteMDP[State, Action],
    gamma: float,
    tol: float,
    maxiter: int = 100,
) -> int:
    """Performs value iteration to evolve the supplied state value mapping `v`
    into state values for an optimal policy.

    Args:
      v: initial state value estimates that will be updated in place
      mdp: FiniteMDP for which optimal state values are to be estimated
      gamma: discount factor
      tol: if the maximum absolute change of state values in one sweep is
        below this value, then iteration will stop
      maxiter: if the number of sweeps of value iteration reaches this value
        then iteration will stop

    Returns:
      niter: number of sweeps of the state space that took place
    """
    for niter in tqdm(range(1, maxiter + 1)):
        delta_v = 0.0
        for s in mdp.states:
            v_old = v[s]
            _, v[s] = backup_single_state_optimal_actions(mdp, s, v, gamma)
            delta_v = max(delta_v, abs(v_old - v[s]))
        if delta_v < tol:
            break
    else:
        warn("`maxiter` reached without convergence within tolerance `tol`")
    return niter
