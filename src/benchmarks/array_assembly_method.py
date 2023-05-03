from itertools import count, repeat
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


def setup_array_initialized(array_type: str, axis_size: int, cells_to_fill: int) -> str:
    initialization = f"initialised_array = {array_type:s}((data, (rows, cols)), shape=({axis_size:d}, {axis_size:d}))"
    base_setup = array_setup(array_type, axis_size, cells_to_fill).split("\n")
    return "\n".join([*base_setup, initialization]).lstrip()


_T = TypeVar("_T")


def analyze_array(stmt: str, setup: str, extractor: Callable, retrieve_stmt=None):
    exec(setup)
    if retrieve_stmt:
        exec(stmt)
        array = eval(retrieve_stmt)
    else:
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
    # Parameter configurations.
    range_axis = np.arange(2048, 16_384 + 1, 2048)
    range_density = np.arange(0, 1.1, 0.1)
    filename_suffix = "size_density"
    # Process parameter configurations for benchmarks.
    repeat_axes = np.repeat(range_axis, len(range_density))
    array_sizes = repeat_axes ** 2
    cells_to_fill = array_sizes * np.tile(range_density, len(range_axis))
    parameters = tuple(zip(repeat_axes, np.floor(cells_to_fill).astype(int)))
    # Run benchmarks.
    # construct_coo_init(*parameters, filename_suffix=filename_suffix)
    # construct_csc_init(*parameters, filename_suffix=filename_suffix)
    # construct_csr_init(*parameters, filename_suffix=filename_suffix)
    # convert_coo_csc(*parameters, filename_suffix=filename_suffix)
    # convert_coo_csr(*parameters, filename_suffix=filename_suffix)
    convert_csc_coo(*parameters, filename_suffix=filename_suffix)
    convert_csr_coo(*parameters, filename_suffix=filename_suffix)


def benchmark_array_size_blowup() -> None:
    range_axis = np.arange(2 ** 20, (2 ** 30 - 1) + (2 ** 22), 2 ** 22)
    cells_to_fill = 2 ** 20
    filename_suffix = "size_same_density"
    parameters = tuple(zip(range_axis, repeat(cells_to_fill)))
    # Run benchmarks
    # construct_coo_init(*parameters, filename_suffix=filename_suffix)
    # construct_csc_init(*parameters, filename_suffix=filename_suffix)
    # construct_csr_init(*parameters, filename_suffix=filename_suffix)
    convert_coo_csc(*parameters, filename_suffix=filename_suffix)
    # convert_coo_csr(*parameters, filename_suffix=filename_suffix)
    convert_csc_coo(*parameters, filename_suffix=filename_suffix)
    convert_csr_coo(*parameters, filename_suffix=filename_suffix)


def generic_func(func: Callable, filename_suffix, *parameters):
    # TODO: Great function name, work in process.
    benchmark_name = f"{func.__name__:s}_{filename_suffix:s}"
    print(f"Benchmarking {benchmark_name} . . .")
    benchmark_data = []
    for parameter_combo in parameters:
        benchmark_data.append(func(*parameter_combo))
    csv_writer(benchmark_name, benchmark_data)


def dummy_func(axis_size, cells_to_fill):
    stmt = "initialised_array.tocsr()"
    setup = setup_array_initialized("csr_array", axis_size, cells_to_fill)
    array_info = analyze_array(stmt, setup, scipy_cs_array_info)
    timings = run_benchmark(stmt, setup, repeats=8)
    return {
            **array_info,
            "axis_size": axis_size,
            "cells_to_fill": cells_to_fill,
            **timings,
    }


def iterate_dok(*args: tuple[int, int], filename_suffix: str) -> None:
    stmt = dedent(
        """
        transition_matrix = dok_array((axis_size, axis_size))
        for index_row, index_col, num in zip(rows, cols, data):
            transition_matrix[index_row, index_col] = num
        """
    )
    benchmark_data = []
    counter = count()
    print(f"iterate_dok_{filename_suffix}")
    for axis_size, cells_to_fill in args:
        print(f"Config {next(counter)}: {axis_size}, {cells_to_fill}")
        setup = array_setup("dok_array", axis_size, cells_to_fill)
        array_info = analyze_array(stmt, setup, None,
                                   retrieve_stmt="transition_matrix")


def iterate_lil(*args: tuple[int, int], filename_suffix: str) -> None:
    # setup = dedent(
    #     """
    # from numpy import array
    # from numpy.random import default_rng
    # from math import pow
    # from scipy.sparse import lil_array
    #
    # axis_size = {axis_size:d}
    # axis_range = array(range(0, axis_size))
    # cells_count = int(pow(axis_size, 2) * {density:f})
    # rows = default_rng(1).choice(axis_range, cells_count)
    # cols = default_rng(1).choice(axis_range, cells_count)
    # data = default_rng(1).random(cells_count)
    # """
    # )
    stmt = dedent(
        """
        transition_matrix = lil_array((axis_size, axis_size))
        for index_row, index_col, num in zip(rows, cols, data):
            transition_matrix[index_row, index_col] = num
        """
    )
    benchmark_data = []
    counter = count()
    print(f"iterate_lil_{filename_suffix}")
    for axis_size, cells_to_fill in args:
        print(f"Config {next(counter)}: {axis_size}, {cells_to_fill}")
        setup = array_setup("dok_array", axis_size, cells_to_fill)
        array_info = analyze_array(stmt, setup, None, retrieve_stmt="transition_matrix")
    # timer = Timer(stmt, setup)
    # data = timer.repeat(10, 10)
    # print(data)


def construct_coo_init(*args: tuple[int, int], filename_suffix: str) -> None:
    stmt = "coo_array((data, (rows, cols)), shape=(axis_size, axis_size))"
    benchmark_data = []
    counter = count()
    print(f"construct_coo_init_{filename_suffix}")
    for axis_size, cells_to_fill in args:
        print(f"Config {next(counter)}: {axis_size}, {cells_to_fill}")
        setup = array_setup("coo_array", axis_size, cells_to_fill)
        array_info = analyze_array(stmt, setup, scipy_coo_array_info)
        timings = run_benchmark(
            stmt,
            setup,
            repeats=8,
        )
        benchmark_data.append(
            {
                **array_info,
                "axis_size": axis_size,
                "cells_to_fill": cells_to_fill,
                **timings,
            }
        )
    csv_writer(f"construct_coo_init_{filename_suffix}", benchmark_data)


def construct_csc_init(*args: tuple[int, int], filename_suffix: str) -> None:
    stmt = "csc_array((data, (rows, cols)), shape=(axis_size, axis_size))"
    benchmark_data = []
    counter = count()
    print(f"construct_csc_init_{filename_suffix}")
    for axis_size, cells_to_fill in args:
        print(f"Config {next(counter)}: {axis_size}, {cells_to_fill}")
        setup = array_setup("csc_array", axis_size, cells_to_fill)
        array_info = analyze_array(stmt, setup, scipy_cs_array_info)
        timings = run_benchmark(stmt, setup, repeats=8)
        benchmark_data.append(
            {
                **array_info,
                "axis_size": axis_size,
                "cells_to_fill": cells_to_fill,
                **timings,
            }
        )
    csv_writer(f"construct_csc_init_{filename_suffix}", benchmark_data)


def construct_csr_init(*args: tuple[int, int], filename_suffix: str) -> None:
    stmt = "csr_array((data, (rows, cols)), shape=(axis_size, axis_size))"
    benchmark_data = []
    counter = count()
    print(f"construct_csr_init_{filename_suffix}")
    for axis_size, cells_to_fill in args:
        print(f"Config {next(counter)}: {axis_size}, {cells_to_fill}")
        setup = array_setup("csr_array", axis_size, cells_to_fill)
        array_info = analyze_array(stmt, setup, scipy_cs_array_info)
        timings = run_benchmark(stmt, setup, repeats=8)
        benchmark_data.append(
            {
                **array_info,
                "axis_size": axis_size,
                "cells_to_fill": cells_to_fill,
                **timings,
            }
        )
    csv_writer(f"construct_csr_init_{filename_suffix}", benchmark_data)


def convert_coo_csc(*args: tuple[int, int], filename_suffix: str) -> None:
    stmt = "initialised_array.tocsc()"
    benchmark_data = []
    counter = count()
    print(f"construct_csc_init_{filename_suffix}")
    for axis_size, cells_to_fill in args:
        print(f"Config {next(counter)}: {axis_size}, {cells_to_fill}")
        setup = setup_array_initialized("coo_array", axis_size, cells_to_fill)
        array_info = analyze_array(stmt, setup, scipy_cs_array_info)
        timings = run_benchmark(stmt, setup, repeats=8)
        benchmark_data.append(
            {
                **array_info,
                "axis_size": axis_size,
                "cells_to_fill": cells_to_fill,
                **timings,
            }
        )
    csv_writer(f"convert_coo_csc_{filename_suffix}", benchmark_data)


def convert_coo_csr(*args: tuple[int, int], filename_suffix: str) -> None:
    stmt = "initialised_array.tocsr()"
    benchmark_data = []
    counter = count()
    print(f"construct_csr_init_{filename_suffix}")
    for axis_size, cells_to_fill in args:
        print(f"Config {next(counter)}: {axis_size}, {cells_to_fill}")
        setup = setup_array_initialized("coo_array", axis_size, cells_to_fill)
        array_info = analyze_array(stmt, setup, scipy_cs_array_info)
        timings = run_benchmark(stmt, setup, repeats=8)
        benchmark_data.append(
            {
                **array_info,
                "axis_size": axis_size,
                "cells_to_fill": cells_to_fill,
                **timings,
            }
        )
    csv_writer(f"convert_coo_csr_{filename_suffix}", benchmark_data)


def convert_csc_coo(*args: tuple[int, int], filename_suffix: str) -> None:
    stmt = "initialised_array.tocoo()"
    benchmark_data = []
    counter = count()
    print(f"convert_csc_coo_{filename_suffix}")
    for axis_size, cells_to_fill in args:
        print(f"Config {next(counter)}: {axis_size}, {cells_to_fill}")
        setup = setup_array_initialized("csc_array", axis_size, cells_to_fill)
        array_info = analyze_array(stmt, setup, scipy_coo_array_info)
        timings = run_benchmark(stmt, setup, repeats=8)
        benchmark_data.append(
            {
                **array_info,
                "axis_size": axis_size,
                "cells_to_fill": cells_to_fill,
                **timings,
            }
        )
    csv_writer(f"convert_csc_coo_{filename_suffix}", benchmark_data)


def convert_csr_coo(*args: tuple[int, int], filename_suffix: str) -> None:
    stmt = "initialised_array.tocoo()"
    benchmark_data = []
    counter = count()
    print(f"convert_csr_coo_{filename_suffix}")
    for axis_size, cells_to_fill in args:
        print(f"Config {next(counter)}: {axis_size}, {cells_to_fill}")
        setup = setup_array_initialized("csr_array", axis_size, cells_to_fill)
        array_info = analyze_array(stmt, setup, scipy_coo_array_info)
        timings = run_benchmark(stmt, setup, repeats=8)
        benchmark_data.append(
            {
                **array_info,
                "axis_size": axis_size,
                "cells_to_fill": cells_to_fill,
                **timings,
            }
        )
    csv_writer(f"convert_csr_coo_{filename_suffix}", benchmark_data)


if __name__ == "__main__":
    benchmark_array_construction()
    benchmark_array_size_blowup()
