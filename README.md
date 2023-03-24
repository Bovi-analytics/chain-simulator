Chain simulator
===============

These are the humble beginnings of a generic and highly scalable platform 
for simulating digital twins using Markov chains.



# Contributing
The package is written in [Python](https://www.python.org/), specifically for 
Python 3.10 and newer. It is assumed that a supported Python interpreter is 
already installed. Dependency management is done using [Poetry](https://python-poetry.org/), 
make sure this tool is installed too. To set up your development environment, 
do the following:

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
    poetry install
    ```

You should now be set up for contributing to the simulation platform!



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
```



------------------------
Author: Max Nollet  
Last updated: 21-03-2023
