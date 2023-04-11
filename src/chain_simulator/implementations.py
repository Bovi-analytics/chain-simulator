from decimal import Decimal
from itertools import count, product
from typing import Generator, TypeVar

from scipy.sparse import csc_array, csr_array, lil_array
from typing_extensions import Self

from chain_simulator.abstract import (
    AbstractArrayAssemblerV1,
)


class ScipyCSRAssembler(AbstractArrayAssemblerV1[csr_array]):
    def assemble(self: Self) -> csr_array:
        calculator = self.probability_calculator
        states = calculator.states
        allocated_array = self.allocate_array(len(states))
        state_index = self.states_to_index(states)
        for state_from, state_to in self.state_combinations(states):
            probability = calculator.probability(state_from, state_to)
            if probability > Decimal("0"):
                index_row = state_index[state_from]
                index_col = state_index[state_to]
                allocated_array[index_row, index_col] = probability
        return allocated_array.tocsr()

    @classmethod
    def allocate_array(cls, size: int, dtype: str = "float64") -> lil_array:
        return lil_array((size, size), dtype=dtype)

    @classmethod
    def states_to_index(cls, states: tuple[str, ...]) -> dict[str, int]:
        index = {}
        for state, number in zip(states, count()):
            index[state] = number
        return index

    @classmethod
    def state_combinations(
        cls, states: tuple[str, ...]
    ) -> Generator[tuple[str, str], None, None]:
        yield from product(states, repeat=2)


_T = TypeVar("_T", csr_array, csc_array)


def chain_simulator(array: _T, steps: int) -> _T:
    if steps == 1:
        return array @ array
    new_array = array
    for _step in range(steps):
        new_array = array @ new_array
    return new_array
