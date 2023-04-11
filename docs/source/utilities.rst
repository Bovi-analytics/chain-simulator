=========
Utilities
=========


----------------------------
Transition matrix validation
----------------------------

There are currently two helper-functions to validate a transition matrix:
:func:`~chain_simulator.utilities.validate_matrix_sum` and
:func:`~chain_simulator.utilities.validate_matrix_positive`. Both functions
return a boolean indicating whether a transition matrix is valid or not.

First we make the necessary imports:

.. doctest::

   >>> from chain_simulator.utilities import validate_matrix_sum, validate_matrix_positive
   >>> import numpy as np

Then we create an array and validate it:

.. doctest::

   >>> valid_matrix = np.array(
   ...    [
   ...       [0.0, 1.0, 0.0],
   ...       [0.0, 0.5, 0.5],
   ...       [0.0, 0.0, 1.0],
   ...    ]
   ... )
   >>> validate_matrix_sum(valid_matrix)
   True
   >>> validate_matrix_positive(valid_matrix)
   True


Logging errors
--------------

It may be helpful to know which row doesn't sum to 1 or which probability is
not positive. This information is logged to the console using the
`logging module <https://docs.python.org/3/library/logging.html>`_ from the
Python standard library. It is disabled by default but can be enabled by adding
the following statements to your code:

.. doctest::

   >>> import logging
   >>> logging.basicConfig()

If we were to validate a faulty transition matrix:

.. doctest::

   >>> faulty_sum_matrix = np.array([[-1, 1, 0], [1, 0, -1], [0, -1, 1]])
   >>> validate_matrix_sum(faulty_sum_matrix)
   False

Then errors that have been found will be logged: ::

   WARNING:chain_simulator.utilities:Row 0 sums to 0 instead of 1!
   WARNING:chain_simulator.utilities:Row 1 sums to 0 instead of 1!
   WARNING:chain_simulator.utilities:Row 2 sums to 0 instead of 1!
