from textwrap import dedent
from timeit import Timer
from typing import Callable, Iterable, Optional, TypeVar

import numpy as np

from benchmarks.abstract import csv_writer
from benchmarks.tools import scipy_coo_array_info, scipy_cs_array_info


def array_setup(array_type: str, axis_size: int, cells_to_fill: int) -> str:
    return dedent(
        f"""
        from math import pow

        import numpy as np
        from scipy.sparse import {array_type:s}
        
        rng = np.random.default_rng(1)
    
        axis_size = {axis_size:d}
        cells_to_fill = {cells_to_fill:d}
        if cells_to_fill:
            array_size = int(pow(axis_size, 2))
            cells = rng.choice(array_size, size=cells_to_fill, replace=False)
            rows, cols = np.divmod(cells, axis_size)
            data = rng.random(cells_to_fill)
        else:
            rows, cols, data = [], [], []
        """
    )


_T = TypeVar("_T")


def analyze_array(stmt: str, setup: str, extractor: Callable):
    exec(setup)
    array = eval(stmt)
    return extractor(array)


def name_timings(timings: Iterable[_T]) -> dict[str, _T]:
    return {f"run_{run:02d}": timing for run, timing in enumerate(timings)}


def run_benchmark(
        stmt: str, setup: str, number: Optional[int] = None, repeats: int = 10
) -> dict[str, float]:
    timer = Timer(stmt, setup)
    if not number:
        autorange = timer.autorange()
        number = autorange[0] if autorange[0] > 2 else 2
    timings = timer.repeat(repeats, number)
    return {
        "number_of_executions": number,
        "number_of_repeats": repeats,
        **name_timings(timings)
    }


def benchmark_array_construction() -> None:
    # Parameter configurations. 16_384
    range_axis = np.arange(1_024, 16_384 + 1, 1024)
    range_density = np.arange(0, 1.05, 0.05)
    # Process parameter configurations for benchmarks.
    repeat_axes = np.repeat(range_axis, len(range_density))
    array_sizes = repeat_axes ** 2
    cells_to_fill = array_sizes * np.tile(range_density, len(range_axis))
    parameters = tuple(zip(repeat_axes, np.floor(cells_to_fill).astype(int)))
    # Run benchmarks.
    construct_coo_init(*parameters)
    construct_csc_init(*parameters)
    construct_csr_init(*parameters)


def iterate_lil() -> None:
    setup = dedent(
        """
    from numpy import array
    from numpy.random import default_rng
    from math import pow
    from scipy.sparse import lil_array

    axis_size = {axis_size:d}
    axis_range = array(range(0, axis_size))
    cells_count = int(pow(axis_size, 2) * {density:f})
    rows = default_rng(1).choice(axis_range, cells_count)
    cols = default_rng(1).choice(axis_range, cells_count)
    data = default_rng(1).random(cells_count)
    """
    )
    stmt = dedent(
        """
        array = lil_array((axis_size, axis_size), dtype="float64")
        for index_row, index_col, num in zip(rows, cols, data):
            array[index_row, index_col] = num
        """
    )
    timer = Timer(stmt, setup)
    data = timer.repeat(10, 10)
    print(data)


def construct_csr_init(*args: tuple[int, int]) -> None:
    stmt = "csr_array((data, (rows, cols)), shape=(axis_size, axis_size))"
    benchmark_data = []
    for axis_size, cells_to_fill in args:
        setup = array_setup("csr_array", axis_size, cells_to_fill)
        array_info = analyze_array(stmt, setup, scipy_cs_array_info)
        timings = run_benchmark(stmt, setup, repeats=10)
        benchmark_data.append(
            {
                **array_info,
                "axis_size": axis_size,
                "cells_to_fill": cells_to_fill,
                **timings,
            }
        )
    csv_writer("construct_csr_init", benchmark_data)


def construct_csc_init(*args: tuple[int, int]) -> None:
    stmt = "csc_array((data, (rows, cols)), shape=(axis_size, axis_size))"
    benchmark_data = []
    for axis_size, cells_to_fill in args:
        setup = array_setup("csc_array", axis_size, cells_to_fill)
        array_info = analyze_array(stmt, setup, scipy_cs_array_info)
        timings = run_benchmark(stmt, setup, repeats=10)
        benchmark_data.append(
            {
                **array_info,
                "axis_size": axis_size,
                "cells_to_fill": cells_to_fill,
                **timings,
            }
        )
    csv_writer("construct_csc_init", benchmark_data)


def construct_coo_init(*args: tuple[int, int]) -> None:
    stmt = "coo_array((data, (rows, cols)), shape=(axis_size, axis_size))"
    benchmark_data = []
    for axis_size, cells_to_fill in args:
        setup = array_setup("coo_array", axis_size, cells_to_fill)
        array_info = analyze_array(stmt, setup, scipy_coo_array_info)
        timings = run_benchmark(
            stmt,
            setup,
            repeats=10,
        )
        benchmark_data.append(
            {
                **array_info,
                "axis_size": axis_size,
                "cells_to_fill": cells_to_fill,
                **timings,
            }
        )
    csv_writer("construct_coo_init", benchmark_data)


if __name__ == "__main__":
    benchmark_array_construction()
