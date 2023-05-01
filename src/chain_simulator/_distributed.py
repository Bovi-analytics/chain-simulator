from typing import Callable, Concatenate, Iterable, ParamSpec, TypeVar

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
    **kwargs: _P.kwargs,
) -> _U:
    state_bag = db.from_sequence(  # type:ignore[attr-defined]
        all_states, partition_size=partition_size
    )
    pending_probabilities = state_bag.map(calculator, *args, **kwargs)
    if flatten:
        pending_probabilities = pending_probabilities.flatten().compute()
    return pending_probabilities.compute()  # type:ignore[no-any-return]


def dummy(state: int, cow_obj: str) -> tuple[tuple[int, int, float], ...]:
    return (1, 1, 0.69), (2, 3, 0.4)
