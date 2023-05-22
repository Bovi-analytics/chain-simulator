"""
====================================================
Simulation tools (:mod:`chain_simulator.simulation`)
====================================================.

.. currentmodule:: chain_simulator.simulation

Chain simulator simulation tools for Markov chains.

Contents
========

.. autosummary::

   state_vector_processor
   vector_processor_numpy
   vector_processor_scipy
   vector_processor_cupy
"""


__all__ = [
    "state_vector_processor",
    "vector_processor_numpy",
    "vector_processor_scipy",
    "vector_processor_cupy",
]

from chain_simulator._simulation import (
    state_vector_processor,
    vector_processor_cupy,
    vector_processor_numpy,
    vector_processor_scipy,
)
