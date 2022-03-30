from dp.jackscarrental import JacksCarRental
from tests.test_jackscarrental import TestJacksCarRentalFunctional


if __name__ == "__main__":
    jcr = JacksCarRental(
        capacity=20,
        overnight_moves_limit=5,
        exp_demand_per_location=(3, 4),
        exp_returns_per_location=(3, 2),
        reward_for_rental=10.0,
        reward_per_car_for_moving_cars=-2.0,
    )
    ts = TestJacksCarRentalFunctional()
    ts.test_policy_iteration_yields_textbook_solution(jcr)
