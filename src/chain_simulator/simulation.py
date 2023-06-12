"""Core simulation functionality.

This module provides the core functionality for simulating digital twins
using Markov Chains. There are both high- and low-level functions
available. The high-level functions automatically makes use of optimal
low-level functions, based on input parameters. The low-level functions
are also available in case the high-level functions aren't of any use.
"""

from chain_simulator._simulation import (
    state_vector_processor,
    vector_processor_cupy,
    vector_processor_numpy,
    vector_processor_scipy,
)

__all__ = [
    "state_vector_processor",
    "vector_processor_numpy",
    "vector_processor_scipy",
    "vector_processor_cupy",
]
