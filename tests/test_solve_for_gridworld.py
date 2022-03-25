import warnings
from typing import List, Tuple
import pytest
from numpy.testing import assert_almost_equal
import numpy as np
from dp.gridworld import GridWorld, GWState, GWAction
from dp import StateValueFunction
from dp.solve import (
    backup_optimal_values,
    policy_evaluation_affine_operator,
    policy_evaluation,
    exact_optimum_state_values,
    iterative_policy_evaluation,
    policy_iteration,
    value_iteration,
)


def pi(s: GWState) -> List[Tuple[GWAction, float]]:
    """Random policy for gridworld."""
    return [(GWAction(a), 0.25) for a in ["n", "e", "w", "s"]]


class TestExactStateValues:
    def test_state_values_for_sutton_barto_gridworld(
        self, gridworld: GridWorld
    ) -> None:
        """Compares outputs of `exact_state_values` against results given
        in the textbook where the specified gridworld is introduced."""
        v = policy_evaluation(gridworld, pi=pi, gamma=0.9)
        assert round(v(GWState((3, 1))), 1) == -0.4
        assert round(v(GWState((4, 3))), 1) == -1.4
        assert round(v(GWState((0, 1))), 1) == 8.8


class TestExactOptimumStateValues:
    def test_optimum_state_values_for_sutton_barto_gridworld(
        self, gridworld: GridWorld
    ) -> None:
        """Compares outputs of `exact_optimum_state_values` against results
        given in the textbook where this gridworld is introduced."""
        v_star = exact_optimum_state_values(gridworld, gamma=0.9)
        assert round(v_star(GWState((3, 1))), 1) == 17.8
        assert round(v_star(GWState((4, 3))), 1) == 13.0
        assert round(v_star(GWState((0, 1))), 1) == 24.4


class TestIterativePolicyEvaluation:
    def test_iterative_policy_evaluation(self, gridworld: GridWorld) -> None:
        v = StateValueFunction(gridworld)
        niter = iterative_policy_evaluation(
            v,
            pi=pi,
            gamma=0.9,
            tol=1e-4,
        )
        assert niter > 0 and niter < 50  # should take less than 50 sweeps
        assert round(v(GWState((3, 1))), 1) == -0.4
        assert round(v(GWState((4, 3))), 1) == -1.4
        assert round(v(GWState((0, 1))), 1) == 8.8

    def test_maxiter_terminates_iteration(self, gridworld: GridWorld) -> None:
        with pytest.warns(UserWarning):
            niter = iterative_policy_evaluation(
                StateValueFunction(gridworld),
                pi=pi,
                gamma=0.9,
                tol=1e-10,
                maxiter=10,
            )
        assert niter == 10

    def test_can_set_tol_to_none_to_perform_fixed_number_of_iterations(
        self, gridworld: GridWorld
    ) -> None:
        # We catch warnings to ensure no warning is raised if tol is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            niter = iterative_policy_evaluation(
                StateValueFunction(gridworld),
                pi=pi,
                gamma=0.9,
                tol=None,
                maxiter=5,
            )
        assert niter == 5


class TestPolicyIteration:
    def test_policy_iteration_functionality(
        self, gridworld: GridWorld
    ) -> None:
        # Set up initial state value estimates and deterministic policy
        v = StateValueFunction(gridworld)
        pi = {s: GWAction("n") for s in gridworld.states}

        # Perform policy iteration to refine both v and pi
        niter = policy_iteration(v, pi, gamma=0.9, tol=1e-4)

        # Check whether policy matches optimal policy given in Sutton-Baro
        # for this gridworld
        assert niter > 0 and niter < 10  # should take less than 10 iterations
        assert pi[GWState((0, 0))] == "e"
        assert pi[GWState((1, 2))] in {"n", "w"}
        assert pi[GWState((4, 1))] == "n"

    def test_maxiter_terminates_iteration(self, gridworld: GridWorld) -> None:
        with pytest.warns(UserWarning):
            niter = policy_iteration(
                StateValueFunction(gridworld),
                {s: GWAction("n") for s in gridworld.states},
                gamma=0.9,
                tol=1e-4,
                maxiter=2,
            )
        assert niter == 2


class TestValueIteration:
    def test_value_iteration(self, gridworld: GridWorld) -> None:
        # Set up initial state values function and iterate
        v_star = StateValueFunction(gridworld)
        niter = value_iteration(v_star, gamma=0.9, tol=1e-4)

        # Check whether state values match optimal values given in
        # Sutton-Barto for this gridworld
        assert niter > 0 and niter < 30  # typically converges in less than 30
        assert round(v_star(GWState((3, 1))), 1) == 17.8
        assert round(v_star(GWState((4, 3))), 1) == 13.0
        assert round(v_star(GWState((0, 1))), 1) == 24.4

    def test_maxiter_terminates_iteration(self, gridworld: GridWorld) -> None:
        v_star = StateValueFunction(gridworld)
        with pytest.warns(UserWarning):
            niter = value_iteration(v_star, 0.9, 1e-14, maxiter=2)
        assert niter == 2


class TestSolverBasicComponents:
    def test_backup_single_state_value(self, gridworld: GridWorld) -> None:
        # Initialise state value mapping arbitrarily as follows
        v = StateValueFunction(
            gridworld,
            initial_value=[gridworld.s2i(s) for s in gridworld.states],
        )

        # Perform a backup update on a single state with arbitrary random
        # policy and check it meets expectations
        fixed_pi = (
            (GWAction("n"), 0.85),
            (GWAction("e"), 0.05),
            (GWAction("w"), 0.05),
            (GWAction("s"), 0.05),
        )
        updated_v = v.backup_state_value(
            GWState((4, 0)), gamma=0.9, pi=lambda s: fixed_pi
        )
        assert gridworld.s2i(GWState((4, 0))) == 20  # sanity check
        expected_v = (
            0.05 * (-1 + 0.9 * 20)  # moving west
            + 0.85 * 0.9 * 15  # moving north
            + 0.05 * 0.9 * 21  # moving east
            + 0.05 * (-1 + 0.9 * 20)  # moving south
        )
        assert_almost_equal(updated_v, expected_v)

    def test_backup_optimal_actions(self, gridworld: GridWorld) -> None:
        # Initialise state value mapping arbitrarily as follows
        v = StateValueFunction(
            gridworld,
            [gridworld.s2i(s) for s in gridworld.states],
        )

        # Call method to identify optimal action and action value estimated
        # for an arbitrarily chosen state and make sure it conforms to
        # expectations
        state = GWState((4, 0))
        assert gridworld.s2i(state) == 20  # sanity check
        actions, action_value = v.backup_optimal_actions(state, gamma=0.9)
        assert len(actions) == 1
        assert actions[0] == GWAction("e")  # RHS worked out by hand
        assert action_value == 0.9 * 21  # corresponding action value

    def test_policy_evaluation_affine_operator(
        self, gridworld: GridWorld
    ) -> None:
        A, b = policy_evaluation_affine_operator(
            gridworld,
            lambda s: [(a, 0.25) for a in gridworld.actions(s)],
            0.9,
        )
        assert b[gridworld.s2i(GWState((1, 2)))] == 0
        assert b[gridworld.s2i(GWState((2, 4)))] == -0.25
        assert b[gridworld.s2i(GWState((0, 3)))] == 5
        assert (
            A[gridworld.s2i(GWState((2, 4))), gridworld.s2i(GWState((2, 4)))]
            == 0.9 * 0.25
        )
        assert (
            A[gridworld.s2i(GWState((1, 2))), gridworld.s2i(GWState((3, 3)))]
            == 0
        )

    def test_backup_optimal_values(self, gridworld: GridWorld) -> None:
        initial_state_values = np.arange(len(gridworld.states), dtype=float)
        backed_up_values = backup_optimal_values(
            gridworld, initial_state_values, 0.9
        )

        # Check values match manual calculation
        assert backed_up_values[7] == 0.9 * 12
        assert backed_up_values[1] == 10 + 0.9 * 21
