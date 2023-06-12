"""Various utilities for validating and processing simulation output.

This module provides various utilities for validating transition matrices and
for processing output from a simulation. Validation utilities can be used to
reduce chances of simulating a faulty transition matrix. The accumulator can be
used to process state vector from a simulation, e.g. to calculate phenotype
data.
"""
from chain_simulator._utilities import (
    AccumulationError,
    CallbackError,
    NoCallbackWarning,
    TransitionMatrixNegativeWarning,
    TransitionMatrixSumWarning,
    simulation_accumulator,
    validate_matrix_negative,
    validate_matrix_sum,
)

__all__ = [
    "AccumulationError",
    "CallbackError",
    "NoCallbackWarning",
    "TransitionMatrixNegativeWarning",
    "TransitionMatrixSumWarning",
    "simulation_accumulator",
    "validate_matrix_negative",
    "validate_matrix_sum",
]
