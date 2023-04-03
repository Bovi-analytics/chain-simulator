from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Generic, TypeVar

from typing_extensions import Self

_T = TypeVar("_T")


class AbstractDigitalTwinFacade(ABC, Generic[_T]):
    """Facade for getting probabilities from a digital twin.

    Abstract class which functions as a facade for a digital twin to get
    probabilities for transitioning from one state to another. This is a
    bridge to 'tell' chain-simulator how to het probabilities from a
    digital twin.

    The method :func:`~AbstractDigitalTwinFacade.probability` is marked as
    abstract. This method needs to be implemented so that simulation-platform
    can convert the digital twin into a transition matrix.
    """

    __slots__ = ("_digital_twin", "_states")

    def __init__(
        self: Self, digital_twin: _T, states: tuple[str, ...]
    ) -> None:
        """Construct the facade.

        Method to construct the facade. The facade is composed of the digital
        twin itself and all states it can be in. This data is later available
        in :func:`~AbstractDigitalTwinFacade.probability`.

        :param digital_twin: Digital twin to get probabilities from.
        :type digital_twin: _T
        :param states: All possible states the digital twin can be in.
        :type states: tuple[str, ...]
        """
        self._digital_twin = digital_twin
        self._states = states

    @property
    def digital_twin(self: Self) -> _T:
        """Digital twin to calculate transition probabilities of.

        :return: Digital twin to get probabilities from.
        :rtype: _T
        """
        return self._digital_twin

    @property
    def states(self: Self) -> tuple[str, ...]:
        """All states a digital twin can be in.

        :return: All states a digital twin can be in.
        :rtype: tuple[str, ...]
        """
        return self._states

    @abstractmethod
    def probability(self: Self, state_from: str, state_to: str) -> Decimal:
        """Calculate the probability of transitioning to a specific state.

        Method to calculate the probability of transitioning from one state
        to another one. This method is marked 'abstract' as it needs to be
        implemented for every type of digital twin.

        :param state_from: State to transition from.
        :type state_from: str
        :param state_to: State to transition to.
        :type state_to: str
        :return: Probability of transitioning from `state_from` to `state_to`.
        :rtype: Decimal
        """
        raise NotImplementedError


_U = TypeVar("_U")


class AbstractArrayAssemblerV1(ABC, Generic[_U]):
    __slots__ = "_probability_calculator"

    def __init__(
        self: Self, probability_calculator: AbstractDigitalTwinFacade[_T]
    ) -> None:
        self._probability_calculator = probability_calculator

    @property
    def probability_calculator(self) -> AbstractDigitalTwinFacade[_T]:
        return self._probability_calculator

    @abstractmethod
    def assemble(self) -> _U:
        raise NotImplementedError
