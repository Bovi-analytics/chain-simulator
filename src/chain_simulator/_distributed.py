from typing import (
    Callable,
    Iterable,
    Optional,
    Protocol,
    TypeVar,
    ParamSpec,
    Concatenate,
)

from itertools import chain

import dask.bag as db

_T = TypeVar("_T")
_U = TypeVar("_U")
_P = ParamSpec("_P")


def state_calculator(
    all_states: Iterable[_T],
    calculator: Callable[Concatenate[_T, _P], _U],
    partition_size: int,
    flatten: bool,
    *args: _P.args,
    **kwargs: _P.kwargs
) -> _U:
    state_bag = db.from_sequence(all_states, partition_size=partition_size)
    pending_probabilities = state_bag.map(calculator, *args, **kwargs)
    if flatten:
        pending_probabilities = pending_probabilities.flatten().compute()
    return pending_probabilities.compute()


def dummy(state: int, cow_obj: str) -> tuple[tuple[int, int, float], ...]:
    return (1, 1, 0.69), (2, 3, 0.4)


def main() -> None:
    states = (1, 2, 3, 4, 5)
    cow = "7"
    probs = state_calculator(states, dummy, 256, False, cow)
    print(probs)


if __name__ == "__main__":
    main()
