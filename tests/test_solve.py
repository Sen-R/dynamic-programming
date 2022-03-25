from typing import Dict
import pytest
from numpy.testing import assert_almost_equal
import numpy as np
from dp._types import Policy
from dp import FiniteMDP
from dp.solve import (
    backup_optimal_values,
    policy_evaluation,
    policy_evaluation_affine_operator,
)
from .conftest import SimpleMDP, TState, TAction


gamma = 0.9


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
    def test_backup_optimal_values(
        self, test_mdp: FiniteMDP, v: Dict[TState, float]
    ) -> None:
        initial_v_array = np.array(list(v.values()))
        updated_v = backup_optimal_values(test_mdp, initial_v_array, gamma)
        expected_v = [0.725, 1.525, 0.0]
        assert_almost_equal(updated_v, expected_v)


class TestPolicyEvaluationByLinearSolve:
    def test_backup_policy_values_operator(
        self,
        test_mdp: SimpleMDP,
        pi: Policy[TState, TAction],
    ) -> None:
        A, b = policy_evaluation_affine_operator(test_mdp, pi, gamma)

        # A should be gamma times the transition matrix
        expected_A = gamma * np.array(
            [[0.0, 0.45, 0.55], [0.75, 0.0, 0.25], [0.0, 0.0, 1.0]]
        )
        assert_almost_equal(A, expected_A)
        # b should be a vector of expected reward per starting state
        expected_b = np.array([0.1, -0.5, 0.0])
        assert_almost_equal(b, expected_b)

    def test_policy_evaluation(
        self, test_mdp: SimpleMDP, pi: Policy[TState, TAction]
    ) -> None:
        """Tests exact policy evaluation solver against previously
        calculated state values for this policy and MDP."""
        v_known = [-0.14106313, -0.59521761, 0.0]
        v = policy_evaluation(test_mdp, pi, gamma)
        assert_almost_equal(list(v._v.values()), v_known)
