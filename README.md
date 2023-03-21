DigitalCowSimulationPlatform
============================

These are the humble beginnings of a generic and highly scalable platform 
for simulating digital twins using Markov chains.



# Contributing
The package is written in [Python](https://www.python.org/), specifically for 
Python 3.10 and newer. It is assumed that a supported Python interpreter is 
already installed. To start contributing to this package, do the following:

1. Clone this Git-repository: `git clone https://github.com/Bovi-analytics/DigitalCowSimulationPlatform.git chain_simulator`
2. Move into the cloned repository: `cd chain_simulator`
3. Create a new Python virtual environment: `python3 -m venv ./venv`
4. Activate the virtual environment: `source ./venv/bin/activate`
5. Update pip: `python -m pip install -U pip`
6. Install development dependencies: `pip install -r requirements.txt`
7. Install the package in development-mode: `pip install -e .`

You should now be set for contributing to the simulation platform.



# Repository contents
This is the root of this Git repository. There are multiple folders and files 
to be found, below is a brief description of each in 
[TOML](https://toml.io/en/)-format:

```toml
[folders]
docs = "User guide for the simulation platform"
src = "Source code of the simulation platform package"
tests = "Unit tests to test the simulation platform"

[files]
".gitignore" = "List of files and/or folders that must not be version-contolled"
".pre-commit-config.yml" = "Tasks to execute on each commit"
"pyproject.toml" = "Configurations for the build system, linters, type checkers and testing frameworks"
"README.md" = "Description of this repository"
"requirements.txt" = "Packages needed for development of the simulation platform"
"setup.cfg" = "Metadata and settings/requirements for building the package"
```



------------------------
Author: Max Nollet  
Last updated: 21-03-2023
