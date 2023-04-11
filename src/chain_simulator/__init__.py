"""Chain simulator package.

Generic, highly scalable platform for simulating digital twins using
Markov chains.
"""

from chain_simulator.abstract import AbstractDigitalTwinFacade
from chain_simulator.implementations import ScipyCSRAssembler, chain_simulator
