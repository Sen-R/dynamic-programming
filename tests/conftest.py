from typing import List, Collection, Tuple
import pytest
from dp._types import TransitionsMapping, NextStateProbabilityTable
from dp import FiniteMDP, GridWorld


# Example code from pytest documentation to implement slow marker and
# runslow command line option.
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# Fixture definitions
@pytest.fixture
def gridworld():
    """Constructs the gridworld described in Chapter 3 of Sutton, Barto (2018)
    textbook.
    """
    return GridWorld(
        size=5,
        wormholes={
            (0, 1): ((4, 1), 10),
            (0, 3): ((2, 3), 5),
        },
    )


TState = str
TAction = str


class SimpleMDP(FiniteMDP[TState, TAction]):
    _states = ["A", "B", "C"]
    _actions = ["R", "L"]
    _transitions: TransitionsMapping[TState, TAction] = {
        "A": {
            "R": (("B", 0.75, -1.0), ("C", 0.25, 1.0)),
            "L": (("C", 0.75, 1.0), ("B", 0.25, -1.0)),
        },
        "B": {
            "R": (("C", 0.75, 1.0), ("A", 0.25, -1.0)),
            "L": (("A", 0.75, -1.0), ("C", 0.25, 1.0)),
        },
        "C": {"R": (("C", 1.0, 0.0),), "L": (("C", 1.0, 0.0),)},
    }

    @property
    def states(self) -> List[TState]:
        return self._states

    def actions(self, state: TState) -> List[TAction]:
        return self._actions

    def next_states_and_rewards(
        self, state: TState, action: TAction
    ) -> Tuple[NextStateProbabilityTable, float]:
        transitions: Collection[
            Tuple[TState, float, float]
        ] = self._transitions[state][action]
        ns_ptable = (
            [ns for ns, p, r in transitions],
            [p for ns, p, r in transitions],
        )
        exp_r = sum(p * r for ns, p, r in transitions)
        return ns_ptable, exp_r


@pytest.fixture
def test_mdp() -> SimpleMDP:
    return SimpleMDP()
