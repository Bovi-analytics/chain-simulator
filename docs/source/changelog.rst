=========
Changelog
=========

Versions follow `SemVer <https://semver.org/spec/v2.0.0.html>`_. The next MAJOR
version bump will happen once the library is out of early development stages.

.. towncrier release notes start

0.3.2 (2023-06-24)
==================

Features
--------

- Added mechanism for processing state vectors using callbacks. (`#27
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/27>`_)
- Made parameter `step_size` available for callback functions. (`#28
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/28>`_)


Improved Documentation
----------------------

- Added documentation to `simulation` module. (`#11
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/11>`_)
- Added usage example to `array_assembler`. (`#12
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/12>`_)
- Added documentation to `utilities` module. (`#14
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/14>`_)


Deprecations and Removals
-------------------------

- Removed deprecated utility function `validate_matrix_positive`.


0.3.1 (2023-05-23)
==================

Features
--------

- Improved speed of matrix multiplications, output remains the same. (`#23
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/23>`_)


0.3.0 (2023-05-22)
==================

Bugfixes
--------

- Warnings are now communicated using `warnings.warn` instead of
  `logging.warning`. (`#18
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/18>`_)
- Transition matrices no longer progress one step too far forward in time.
  (`#19
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/19>`_)


Improved Documentation
----------------------

- Transition matrices no longer progress one step too far forward in time.
  (`#19
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/19>`_)


Deprecations and Removals
-------------------------

- Removed deprecated `chain_simulator.abstract.AbstractDigitalTwinFacade`. (`#7
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/7>`_)
- Removed deprecated `chain_simulator.implementations.ScipyCSRAssembler` and
  `chain_simulator.abstract.AbstractArrayAssemblerV1`. (`#8
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/8>`_)
- Deprecated use of `validate_matrix_positive`, use `validate_matrix_negative`
  instead. (`#21
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/21>`_)


0.2.2 (2023-04-21)
==================

Bugfixes
--------

- Adjusted transition matrix validators to be compatible with SciPy COO/CSC/CSR
  formats. (`#1
  <https://github.com/Bovi-analytics/DigitalCowSimulationPlatform/issues/1>`_)
