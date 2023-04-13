"""Implementations of :mod:`~chain_simulator.abstract`."""

from decimal import Decimal
from itertools import count, product
from typing import Generator, Iterable, TypeVar

from scipy.sparse import coo_array, csc_array, csr_array, lil_array
from typing_extensions import Self

from chain_simulator.abstract import (
    AbstractArrayAssemblerV1,
)


class ScipyCSRAssembler(AbstractArrayAssemblerV1[csr_array]):
    """Implementation of AbstractArrayAssemblerV1.

    Concrete class of
    :class:`~chain_simulator.abstract.AbstractArrayAssemblerV1`.
    """

    def assemble(self: Self) -> csr_array:
        """Assemble an array in the Compressed Sparse Row (CSR) format.

        Method to assemble an array in the Compressed Sparse Row (CSR) format.
        This format excels in representing sparse arrays while also allowing
        fast arithmetic operations on the array.

        :return: Transition matrix in Compressed Sparse Row (CSR) format.
        :rtype: csr_array
        """
        calculator = self.probability_calculator
        states = calculator.states
        state_index = self.states_to_index(states)
        rows = []
        cols = []
        data = []
        for state_from, state_to in self.state_combinations(states):
            if (
                probability := calculator.probability(state_from, state_to)
            ) > Decimal("0"):
                rows.append(state_index[state_from])
                cols.append(state_index[state_to])
                data.append(probability)
        count_states = len(states)
        array = coo_array(
            (data, (rows, cols)),
            shape=(count_states, count_states),
            dtype="float64",
        )
        return array.tocsr()

    @classmethod
    def allocate_array(cls, size: int, dtype: str = "float64") -> lil_array:
        """Allocate an editable array in memory.

        Method to allocate an array in memory that can be edited. Not all
        sparse matrix formats allow mutation. For this reason a List of Lists
        (LIL) format is used.

        :param size: The size of the array to allocate in shape (size, size).
        :type size int
        :param dtype: Data type to use for allocating digits in the array.
        :type dtype: str
        :return: Allocated LIL array pf size `size` and dtype `dtype`.
        :rtype lil_array
        """
        return lil_array((size, size), dtype=dtype)

    @classmethod
    def states_to_index(cls, states: Iterable[str]) -> dict[str, int]:
        """Convert a collection of states into an index.

        Method which builds an index of a collection of states. This index can
        be used to find the ordered number of a state. Useful for translating
        a column/row index into the name of a state.

        :param states: Collection of states.
        :type states: Iterable[str]
        :return: Index of states.
        :rtype: dict[str, int]
        """
        index = {}
        for state, number in zip(states, count(), strict=False):
            index[state] = number
        return index

    @classmethod
    def state_combinations(
        cls, states: Iterable[str]
    ) -> Generator[tuple[str, str], None, None]:
        """Generate all possible combinations of states.

        :param states: States to generate combinations of.
        :type states: Iterable[str]
        :return: Combination of states.
        :rtype: Generator[tuple[str, str], None, None]
        """
        yield from product(states, repeat=2)


_T = TypeVar("_T", csr_array, csc_array)


def chain_simulator(array: _T, steps: int) -> _T:
    """Progress a Markov chain forward in time.

    Method which progresses a Markov chain forward in time using a provided
    transition matrix. Based on the `steps` parameter, the transition matrix is
    raised to the power of `steps`. This is done using a matrix multiplication.

    :param array: Transition matrix.
    :type array: _T
    :param steps: Steps in time to progress the simulation.
    :type steps: int
    :return: Transition matrix progressed in time.
    :rtype _T
    """
    if steps == 1:
        return array @ array
    new_array = array
    for _step in range(steps):
        new_array = array @ new_array
    return new_array
