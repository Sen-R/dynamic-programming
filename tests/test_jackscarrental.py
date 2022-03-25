from typing import Iterable, List
from pathlib import Path
import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import poisson  # type: ignore
from dp import StateValueFunction
from dp.jackscarrental import (
    JacksCarRental,
    CarCounts,
    MoveCars,
    counts_after_moving_cars,
    poisson_table,
)
from dp.solve import policy_iteration


@pytest.fixture
def jcr_mini() -> JacksCarRental:
    """Returns a MDP for the Jack's car rental problem, as specified in
    Sutton, Barto (2018) section 4.3, but with changed parameters to keep
    the state space size manageable for testing purposes."""
    mdp = JacksCarRental(
        capacity=2,
        overnight_moves_limit=2,
        exp_demand_per_location=(1, 2),
        exp_returns_per_location=(1, 1),
        reward_for_rental=10.0,
        reward_per_car_for_moving_cars=-2.0,
    )
    return mdp


p_rent = (poisson(mu=1), poisson(mu=2))
p_ret = (poisson(mu=1), poisson(mu=1))


class TestJacksCarRental:
    def test_states_property(self, jcr_mini: JacksCarRental) -> None:
        """States should consist of a tuple of ints, each between 1 and 20,
        representing the number of cars at the first and second location
        respectively."""
        expected_states = {
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        }
        actual_states = set(jcr_mini.states)
        assert expected_states == actual_states

    @pytest.mark.parametrize(
        "state,exp_range",
        [
            (CarCounts((1, 2)), range(-1, 1)),
            (CarCounts((2, 0)), range(0, 3)),
            (CarCounts((1, 1)), range(-1, 2)),
            (CarCounts((0, 2)), range(-2, 1)),
            (CarCounts((0, 0)), range(0, 1)),
            (CarCounts((2, 2)), range(0, 1)),
        ],
    )
    def test_actions_property(
        self,
        jcr_mini: JacksCarRental,
        state: CarCounts,
        exp_range: Iterable[int],
    ) -> None:
        """Actions should be a (possibly negative) int, representing
        the net movement of cars from location 1 to location 2 overnight."""
        expected_actions = set(MoveCars(m) for m in exp_range)
        actual_actions = set(jcr_mini.actions(state))
        assert expected_actions == actual_actions

    @pytest.mark.parametrize(
        "state,exp_range",
        [
            (CarCounts((10, 10)), range(-5, 6)),
            (CarCounts((3, 10)), range(-5, 4)),
            (CarCounts((17, 6)), range(-3, 6)),
        ],
    )
    def test_overnight_moves_limit(
        self, state: CarCounts, exp_range: Iterable[int]
    ) -> None:
        """Tests whether action space is correct when overnight_moves_limit
        is set."""
        jcr_mini_on_limit = JacksCarRental(
            capacity=20,
            overnight_moves_limit=5,
            exp_demand_per_location=(3, 4),
            exp_returns_per_location=(3, 2),
            reward_for_rental=10.0,
            reward_per_car_for_moving_cars=-2.0,
        )
        expected_actions = set(MoveCars(m) for m in exp_range)
        actual_actions = set(jcr_mini_on_limit.actions(state))
        assert expected_actions == actual_actions

    def test_next_states_and_rewards(self, jcr_mini: JacksCarRental) -> None:
        state = CarCounts((1, 2))  # arbitrary state
        action = MoveCars(-1)  # arbitrary action
        morning_counts = CarCounts((2, 1))  # after applying action
        (
            des_evening_counts,
            exp_rentals,
        ) = jcr_mini.evening_counts_and_exp_rentals(morning_counts)
        des_exp_reward = exp_rentals * 10.0 - 2.0
        act_evening_counts, act_exp_reward = jcr_mini.next_states_and_rewards(
            state, action
        )
        assert_almost_equal(act_exp_reward, des_exp_reward)
        assert act_evening_counts[0], des_evening_counts[0]
        assert_almost_equal(act_evening_counts[1], des_evening_counts[1])

    def test_evening_states_and_exp_rentals(
        self, jcr_mini: JacksCarRental
    ) -> None:
        morning_counts = CarCounts((1, 2))
        ec_ptable, exp_rentals = jcr_mini.evening_counts_and_exp_rentals(
            morning_counts
        )

        # All states are possible next states, let's check this first
        assert set(ec_ptable[0]) == set(jcr_mini.states)

        # Probabilities should sum to one
        assert_almost_equal(sum(ec_ptable[1]), 1.0)

    @pytest.mark.parametrize(
        "branch,cars_morning,des_count_probs,des_exp_rentals",
        [
            (
                0,
                1,
                [
                    (1.0 - p_rent[0].cdf(0)) * p_ret[0].pmf(0),
                    (
                        p_rent[0].pmf(0) * p_ret[0].pmf(0)
                        + (1.0 - p_rent[0].cdf(0)) * p_ret[0].pmf(1)
                    ),
                    (
                        p_rent[0].pmf(0) * (1.0 - p_ret[0].cdf(0))
                        + (1.0 - p_rent[0].cdf(0)) * (1.0 - p_ret[0].cdf(1))
                    ),
                ],
                1.0 - p_rent[0].cdf(0),
            ),
        ],
    )
    def test_branch_evening_count_and_exp_rentals(
        self,
        jcr_mini: JacksCarRental,
        branch: int,
        cars_morning: int,
        des_count_probs: List[float],
        des_exp_rentals: float,
    ) -> None:
        (
            act_count_probs,
            act_exp_rentals,
        ) = jcr_mini.branch_evening_count_and_exp_rentals(branch, cars_morning)
        assert_almost_equal(act_count_probs.sum(), 1.0)
        assert_almost_equal(act_count_probs, des_count_probs)
        assert_almost_equal(act_exp_rentals, des_exp_rentals)


@pytest.mark.parametrize(
    "state,action,counts_after_move",
    [
        (CarCounts((1, 2)), MoveCars(-1), CarCounts((2, 1))),
        (CarCounts((1, 1)), MoveCars(1), CarCounts((0, 2))),
    ],
)
def test_counts_after_moving_cars(
    state: CarCounts,
    action: MoveCars,
    counts_after_move: CarCounts,
) -> None:
    assert counts_after_moving_cars(state, action) == counts_after_move


def test_poisson_table() -> None:
    table = poisson_table(2, mu=2.0)
    assert_almost_equal(table.sum(), 1.0)
    desired = [0.13533528, 0.27067057, 0.59399415]
    assert_almost_equal(table, desired)


# The following is a functional test of the MDP, checking whether
# it reproduces the optimal policy illustrated in the textbook


@pytest.fixture
def jcr() -> JacksCarRental:
    return JacksCarRental(
        capacity=20,
        overnight_moves_limit=5,
        exp_demand_per_location=(3, 4),
        exp_returns_per_location=(3, 2),
        reward_for_rental=10.0,
        reward_per_car_for_moving_cars=-2.0,
    )


class TestJacksCarRentalFunctional:
    @pytest.mark.slow
    def test_policy_iteration_yields_textbook_solution(
        self, jcr: JacksCarRental
    ) -> None:
        # Textbook solution (as I can best deduce from the graph) is
        # stored in a text file. We need to read it in first and
        # then compare to the solution derived by policy iteration

        # Start by reading and parsing the expected solution
        # Note the solution is given in "plotting" format, i.e. need
        # to reverse the vertically to put it in "matrix" format
        solution_file = (
            Path(__file__).parent / "data/jcr_optimal_policy.txt"
        ).resolve()
        assert solution_file.exists()
        policy_matrix = []
        with solution_file.open() as f:
            for line in f:
                row_vector = [int(el) for el in line.split()]
                assert len(row_vector) == 21
                policy_matrix.append(row_vector)
        policy_matrix.reverse()
        assert len(policy_matrix) == 21

        # Computer optimal policy by policy iteration
        v = StateValueFunction(jcr)
        pi = {s: jcr.actions(s)[0] for s in jcr.states}
        policy_iteration(v, pi, 0.9, 0.1)

        # Compare the solution with the desired answer
        for s, a in pi.items():
            des_a = MoveCars(policy_matrix[s[0]][s[1]])
            assert a == des_a, (s, a, des_a)
