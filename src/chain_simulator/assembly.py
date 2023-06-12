"""Transition matrix assembly functions.

This module provides functionality for assembling transition matrices.
Resulting transition matrices are built using sparse formats, e.g. the
Coordinate (COO) format.
"""

from chain_simulator._assembly import array_assembler

__all__ = ["array_assembler"]
