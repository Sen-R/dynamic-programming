""" Implementation of Jack's car rental problem as a `FiniteMDP`.

This problem is described in chapter 4 of the textbook Sutton, Barto (2018).
"""

from typing import NewType, Tuple, List, Dict
from itertools import product
from numpy.typing import NDArray
import numpy as np
from scipy.stats import poisson  # type: ignore
from .base import FiniteMDP
from ._types import NextStateProbabilityTable

CarCounts = NewType("CarCounts", Tuple[int, int])
MoveCars = NewType("MoveCars", int)


class JacksCarRental(FiniteMDP[CarCounts, MoveCars]):
    """MDP for Jack's car rental.

    Args:
      capacity: number of cars that can be held at a location
      overnight_moves_limit: max number of cars that can be moved overnight
      exp_demand_per_location: mean daily rental demand at each location
      exp_returns_per_location: mean daily returns at each location
      reward_for_rental: dollar credit received for renting a car out
        (this is $10 in the problem as originally stated)
      reward_per_car_for_moving_cars: (negative) dollar reward for each
        car to be moved from one location to the other overnight (this is
        -$2 in the problem as originally stated)
    """

    def __init__(
        self,
        capacity: int,
        overnight_moves_limit: int,
        exp_demand_per_location: Tuple[int, int],
        exp_returns_per_location: Tuple[int, int],
        reward_for_rental: float,
        reward_per_car_for_moving_cars: float,
    ):
        self.capacity = capacity
        self.overnight_moves_limit = overnight_moves_limit
        self.exp_demand_per_location = exp_demand_per_location
        self.exp_returns_per_location = exp_returns_per_location
        self.reward_for_rental = reward_for_rental
        self.reward_per_car_for_moving_cars = reward_per_car_for_moving_cars

        # Event space arrays, for computing probabilities
        self.p = (
            np.outer(
                poisson_table(self.capacity, exp_demand_per_location[0]),
                poisson_table(self.capacity, exp_returns_per_location[0]),
            ),
            np.outer(
                poisson_table(self.capacity, exp_demand_per_location[1]),
                poisson_table(self.capacity, exp_returns_per_location[1]),
            ),
        )
        self.demand = np.arange(self.capacity + 1)[:, np.newaxis]
        self.returns = np.arange(self.capacity + 1)[np.newaxis, :]

        # Cache to speed up next states and rewards calculation
        self._cache: Dict[
            CarCounts, Tuple[NextStateProbabilityTable[CarCounts], float]
        ] = {}

    @property
    def states(self) -> List[CarCounts]:
        return [
            CarCounts(t)
            for t in product(
                range(self.capacity + 1), range(self.capacity + 1)
            )
        ]

    def actions(self, state: CarCounts) -> List[MoveCars]:
        min_move = -min(
            self.capacity - state[0], state[1], self.overnight_moves_limit
        )
        max_move = min(
            self.capacity - state[1], state[0], self.overnight_moves_limit
        )
        return [MoveCars(m) for m in range(min_move, max_move + 1)]

    def next_states_and_rewards(
        self, state: CarCounts, action: MoveCars
    ) -> Tuple[NextStateProbabilityTable[CarCounts], float]:
        next_morning_counts = counts_after_moving_cars(state, action)
        assert max(next_morning_counts) <= self.capacity, (state, action)
        assert min(next_morning_counts) >= 0, (state, action)
        next_evening_counts, exp_rentals = self.evening_counts_and_exp_rentals(
            next_morning_counts
        )
        exp_reward = self.exp_reward(action, exp_rentals, next_morning_counts)
        return next_evening_counts, exp_reward

    def exp_reward(
        self,
        action: MoveCars,
        exp_rentals: float,
        next_morning_counts: CarCounts,
    ) -> float:
        """Calculates the expected reward under the given scenario.

        The expected reward for the default problem is a function of how
        many cars were moved (`action`) and the average number of rentals
        over the next day (`exp_rental`). However, this function also takes
        as a parameter the number of cars at each location at the start
        of the next day (`morning_counts`), so that the class can be
        extended to allow for more complex reward functions (such as the
        modified problem described in Exercise 4.7 of the textbook, where
        the reward is also a function of the number of cars kept overnight).
        """
        action_reward = abs(action) * self.reward_per_car_for_moving_cars
        exp_reward = exp_rentals * self.reward_for_rental + action_reward
        return exp_reward

    def evening_counts_and_exp_rentals(
        self, morning_counts: CarCounts
    ) -> Tuple[NextStateProbabilityTable[CarCounts], float]:
        """Returns probability table for end-of-day car counts as well as
        expected number of rentals over the day, given the number of cars
        available in the morning (post any overnight moves).

        Args:
          morning_counts: the number cars at each location at the start of the
            day

        Returns:
          evening_counts: tuple of lists, the first containing possible
            end-of-day counts and the second containing associated
            probabilities
          exp_rentals: expected number of (successful) rentals over that day
        """
        try:
            return self._cache[morning_counts]
        except KeyError:
            (
                ec_prob_0,
                exp_rentals_0,
            ) = self.branch_evening_count_and_exp_rentals(0, morning_counts[0])
            (
                ec_prob_1,
                exp_rentals_1,
            ) = self.branch_evening_count_and_exp_rentals(1, morning_counts[1])
            evening_counts = (
                self.states,
                [ec_prob_0[i] * ec_prob_1[j] for i, j in self.states],
            )
            exp_rentals = exp_rentals_0 + exp_rentals_1
            self._cache[morning_counts] = (evening_counts, exp_rentals)
            return evening_counts, exp_rentals

    def branch_evening_count_and_exp_rentals(
        self, branch: int, cars_morning: int
    ) -> Tuple[NDArray[np.float_], float]:
        """Returns probabilities for all possible end-of-day car counts,
        and expected number of rentals, both conditional on the number
        of cars in the morning, for the specified branch.

        Args:
          branch: index of the branch for which this is being calculated
          cars_morning: number of cars at this branch at the start of the day

        Returns:
          count_probs: array of length `self.capacity + 1` containing
            probability of ending up with 0 to `self.capacity` cars at the
            end of the day, conditional on having started the day with
            `cars_morning` cars
          exp_rentals: expected number of rentals over the day conditional
            on having started the day with `cars_morning` cars
        """
        # Evaluate desired random variables over event space
        cars_hired = np.minimum(cars_morning, self.demand)
        cars_evening = np.minimum(
            self.capacity, cars_morning - cars_hired + self.returns
        )

        # Evaluate desired marginal and expectation (conditional on
        # cars_morning) and return
        count_probs = np.bincount(
            cars_evening.ravel(), weights=self.p[branch].ravel()
        ).astype(np.float_)
        exp_rentals = np.sum(cars_hired * self.p[branch])
        return count_probs, exp_rentals


def counts_after_moving_cars(
    initial_counts: CarCounts,
    move: MoveCars,
) -> CarCounts:
    """Returns car counts in the morning after `move` cars were moved
    overnight.

    Args:
      initial_counts: number of cars at each location prior to any overnight
      moves
      move: number of cars to move from location 0 to location 1 (negative
        number means moving cars in the opposite direction)

    Returns:
      Number of cars at each location the following morning, after overnight
      moves have taken place
    """
    return CarCounts((initial_counts[0] - move, initial_counts[1] + move))


def poisson_table(max_k: int, mu: float) -> NDArray[np.float_]:
    """Returns table of Poisson probabilities up to `k >= max_k`.

    Suppose K is Poisson distributed with mean `mu`. Then this function
    returns an array of length `max_k + 1`, where the first `max_k` entries
    correspond to the probabilities `P(K=0), P(K=1), ..., P(K=max_k-1)` while
    the final entry in the table corresponds to `P(K>=max_k)`.

    Note, this means that the sum of the entries of the array should be 1.

    Args:
      max_k: the ceiling at which `K` is capped
      mu: the expected value of the (uncapped) Poisson distribution

    Returns:
      table: array of length `max_k + 1` with entries as described above
    """
    table = np.zeros(max_k + 1)
    table[:max_k] = poisson.pmf(range(max_k), mu=mu)
    table[-1] = 1.0 - poisson.cdf(max_k - 1, mu=mu)
    return table
