from collections import deque
from functools import partial
from textwrap import dedent
from timeit import Timer

from inspect import getsource, getsourcelines
from typing import Callable, Iterable

from numpy.random import default_rng
from scipy.sparse import csr_array, random
from numpy import arange


def blah():
    density_step = 0.05
    density_range = arange(0, 1 + density_step, density_step)
    stmt = "array.dot(array)"
    args = {"shape": (1024, 1024), "dtype": "float64"}
    # partial_assembly = partial(
    #     assemble_csr_array, shape=(20, 20), dtype="float64"
    # )
    imports = [
        "from numpy.random import default_rng",
        "from scipy.sparse import csr_array, random"
    ]
    for density in density_range:
        setup = build_setup(imports, assemble_csr_array, **args, density=density)
        # print(setup)
        # setup = partial_assembly(density=density)

        timer = Timer(stmt, setup)
        timings = timer.repeat(repeat=5, number=10)
        print(f"{density}, {timings}")

        # print(setup)

        # print("".join(func_lines))
        # print(assemble_csr_array(**args, density=0.5))


def build_setup(imports: Iterable[str], function: Callable, **args) -> str:
    setup = [
        f"from __main__ import {function.__name__}\n",
        "from numpy.random import default_rng\n",
        "from scipy.sparse import csr_array, random\n"
    ]
    func_lines = getsourcelines(function)
    setup.extend(func_lines[0])
    args_converted = args_to_str(**args)
    setup.append(f"array = {function.__name__}({args_converted})")
    return "".join(setup)


def args_to_str(**args):
    converted_pairs = []
    for key, value in args.items():
        if isinstance(value, str):
            converted_pairs.append(f"{key}='{value}'")
        else:
            converted_pairs.append(f"{key}={value}")
    return ", ".join(converted_pairs)


def assemble_csr_array(
        shape: tuple[int, int], density: float, dtype: str
) -> csr_array:
    array = csr_array(
        random(
            shape[0],
            shape[1],
            density=density,
            format='csr',
            dtype=dtype,
            random_state=default_rng(1)
        )
    )
    return array


if __name__ == "__main__":
    blah()
