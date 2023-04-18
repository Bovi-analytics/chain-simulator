Chain simulator
===============

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
----

These are the humble beginnings of a generic and highly scalable platform 
for simulating digital twins using Markov chains.



# Contributing
The package is written in [Python](https://www.python.org/), specifically for 
Python 3.10 and newer. It is assumed that a supported Python interpreter is 
already installed. Dependency management is done using 
[Hatch](https://hatch.pypa.io/latest/), make sure this tool is installed too. 
To set up your development environment, do the following:

1. Clone the chain-simulator repository to your computer:
    ```shell
    git clone https://github.com/Bovi-analytics/DigitalCowSimulationPlatform.git chain_simulator
    ```
2. Move into the new folder named chain_simulator:
    ```shell
    cd chain_simulator
    ```
3. Install all package dependencies in a virtual environment:
    ```shell
    hatch env create
    ```

You should now be set up for contributing to the simulation platform!



# Repository contents
This is the root of this Git repository. There are multiple folders and files 
to be found, below is a brief description of each in 
[TOML](https://toml.io/en/)-format:

```toml
[folders]
".github" = "Mostly GitHub Actions configurations to run tests on multiple operating systems"
docs = "User guide and API documentation for the simulation platform"
src = "Source code of the simulation platform package"
tests = "Unit tests to test the simulation platform"

[files]
".gitignore" = "List of files and/or folders that must not be version-contolled"
".pre-commit-config.yml" = "Tasks to execute on each commit"
pyproject.toml = "Configurations for the build system, linters, type checkers and testing frameworks"
README.md = "Description of this repository"
tox.ini = "Configuration for Tox to run tests on multiple Python versions"
```



------------------------
Author: Max Nollet  
Last updated: 18-04-2023
