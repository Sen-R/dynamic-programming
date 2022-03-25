from typing import Optional, MutableMapping, Tuple
from warnings import warn
from tqdm import tqdm  # type: ignore
from numpy.typing import NDArray
import numpy as np
from scipy import optimize  # type: ignore
from ._types import State, Action, Policy
from .base import FiniteMDP
from .state_values import StateValueFunction


def policy_evaluation_affine_operator(
    mdp: FiniteMDP[State, Action],
    pi: Policy[State, Action],
    gamma: float,
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


def policy_evaluation(
    mdp: FiniteMDP[State, Action], pi: Policy[State, Action], gamma: float
) -> StateValueFunction[State, Action]:
    """Returns state values for given policy.

    This function directly solves the (linear) Bellman equation to calculate
    the state value function for the policy represented by `pi`.

    Args:
      mdp: FiniteMDP for which state values are being calculated
      pi: conditional probabilities for actions given states, encoding the
        policy to be evaluated
      gamma: discount factor

    Returns:
      State value function obtained by solving the Bellman equation
    """
    A, b = policy_evaluation_affine_operator(mdp, pi, gamma)
    v = np.linalg.solve(np.eye(len(mdp.states)) - A, b)
    return StateValueFunction(mdp, v)


def exact_optimum_state_values(
    mdp: FiniteMDP[State, Action], gamma: float, tol: Optional[float] = None
) -> StateValueFunction:
    """Returns state values for an optimal policy for the given MDP.

    This function uses a non-linear solver to directly solve the Bellman
    optimality equation, to calculate state values for an optimal policy.

    Args:
      mdp: FiniteMDP for which optimal state values are being calculated
      gamma: discount factor
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

    return StateValueFunction(mdp, opt_result.x)


def iterative_policy_evaluation(
    v: StateValueFunction[State, Action],
    pi: Policy[State, Action],
    gamma: float,
    tol: Optional[float] = None,
    maxiter: int = 100,
) -> int:
    """Applies iterative policy evaluation to refine provided state value
    estimates.

    Args:
      v: initial estimates of state values which are refined in place
      pi: conditional probabilities for actions given states, encoding the
        policy being evaluated
      gamma: discount factor
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
        for s in v.mdp.states:
            v_old = v._v[s]
            v._v[s] = v.backup_state_value(s, pi, gamma)
            delta_v = max(delta_v, abs(v._v[s] - v_old))
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
    v: StateValueFunction[State, Action],
    pi: MutableMapping[State, Action],
    gamma: float,
    tol: float,
    maxiter: int = 100,
) -> int:
    """Performs policy iteration to refine policy `pi` and corresponding
    state value estimates `v`.

    Args:
      v: Initial estimates of state values for policy `pi`, updated in place
      pi: initial (deterministic) policy, mapping states to actions
      gamma: discount factor
      tol: tolerance for embedded policy evaluation component of each policy
        iteration to converge
      maxiter: maximum number of policy iterations to perform

    Returns:
      niter: number of sweeps of the state space that were performed
    """
    for niter in tqdm(range(1, maxiter + 1)):
        policy_stable = True
        for s in v.mdp.states:
            old_a = pi[s]
            new_as, _ = v.backup_optimal_actions(s, gamma)
            if old_a not in new_as:
                policy_stable = False
                pi[s] = new_as[0]  # Arbitrarily pick one if there are many
        if policy_stable:
            break
        iterative_policy_evaluation(
            v, pi=(lambda s: ((pi[s], 1.0),)), gamma=gamma, tol=tol
        )
    else:
        # maxiter reached but policy not yet stable, so warn
        warn("maxiter reached but policy not yet stable")
    return niter


def value_iteration(
    v: StateValueFunction[State, Action],
    gamma: float,
    tol: float,
    maxiter: int = 100,
) -> int:
    """Performs value iteration to evolve the supplied state value mapping `v`
    into state values for an optimal policy.

    Args:
      v: initial state value estimates that will be updated in place
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
        for s in v.mdp.states:
            v_old = v._v[s]
            _, v._v[s] = v.backup_optimal_actions(s, gamma)
            delta_v = max(delta_v, abs(v_old - v._v[s]))
        if delta_v < tol:
            break
    else:
        warn("`maxiter` reached without convergence within tolerance `tol`")
    return niter
