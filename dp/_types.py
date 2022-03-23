from typing import TypeVar, Collection, Tuple, Mapping, Callable, Sequence


State = TypeVar("State")
Action = TypeVar("Action")
Reward = float
Probability = float
Policy = Callable[[State], Collection[Tuple[Action, Probability]]]
NextStateProbabilityTable = Tuple[Sequence[State], Sequence[Probability]]
TransitionsMapping = Mapping[
    State, Mapping[Action, Collection[Tuple[State, Probability, Reward]]]
]
