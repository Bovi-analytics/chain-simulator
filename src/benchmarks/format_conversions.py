from __future__ import annotations

from datetime import datetime
from math import floor
from pathlib import Path
from textwrap import dedent
from timeit import Timer
from typing import Dict, Generator, List
from dataclasses import dataclass, asdict
from json import dumps
from operator import mul

from numpy import asarray, divmod
from numpy.random import default_rng
from pandas import DataFrame
from scipy import sparse as scipy_sparse


@dataclass
class ConversionMetadata:
    title: str
    run_date: str
    number_of_repeats: int
    number_of_runs: int
    conversions: List[str]
    matrix_shape: List[int]
    nonzero_cells: int


def prepare_array(axis_size: int, cells_to_fill: int, data_type: str):
    rng = default_rng(1)

    axis_size = axis_size
    cells_to_fill = cells_to_fill
    if cells_to_fill:
        array_size = int(pow(axis_size, 2))
        cells = rng.choice(array_size, size=cells_to_fill, replace=False)
        rows, cols = divmod(cells, axis_size)
        data = rng.random(cells_to_fill)
    else:
        rows, cols, data = [], [], []
    return scipy_sparse.coo_array((data, (rows, cols)),
                                  shape=(axis_size, axis_size),
                                  dtype=data_type)


def scipy_converter(
        # shape_m: int,
        # shape_n: int,
        # cells_to_fill: float,
        source_format: str,
        # data_type: str
        matrix_path: Path | str
) -> Generator[Timer, str, None]:
    formats = {
        "bsr": "bsr_array",
        "coo": "coo_array",
        "csc": "csc_array",
        "csr": "csr_array",
        "dia": "dia_array",
        "dok": "dok_array",
        "lil": "lil_array"
    }
    # setup = dedent("""
    #     from __main__ import prepare_array
    #     from scipy.sparse import random, %s, %s
    #
    #     initial_array = %s(prepare_array(%d, %d, '%s'))
    # """).strip()
    # stmt = "converted_array = %s(initial_array)"
    setup = dedent("""
        from scipy.sparse import load_npz, %s, %s
        
        matrix_source = %s(load_npz(r'%s'))
    """).strip()
    stmt = "matrix_target = %s(matrix_source)"
    source_format = formats.get(source_format)
    timer = None
    while True:
        target_format = yield timer
        if converter := formats.get(target_format):
            timer = Timer(
                # setup=setup % (
                #     source_format, converter, source_format, shape_m,
                #     cells_to_fill, data_type
                # ),
                # stmt=stmt % (converter,),
                setup=setup % (converter, source_format, source_format, matrix_path),
                stmt=stmt % converter
            )
        else:
            timer = None


# def cupy_converter(
#         shape_m: int,
#         shape_n: int,
#         cells_to_fill: float,
#         source_format: str,
#         data_type: str
# ) -> Generator[Timer, str, None]:
#     formats = {
#         "coo": "coo_matrix",
#         "csc": "csc_matrix",
#         "csr": "csr_matrix",
#         "dia": "dia_matrix",
#     }
#     setup = dedent("""
#         from __main__ import prepare_array
#         from scipy.sparse import random, %s, %s
#         from cupyx.scipy.sparse import %s
#
#         initial_array = %s(%s(prepare_array(%d, %d, '%s')))
#     """).strip()
#     stmt = "converted_array = %s(initial_array)"
#     cells_to_fill = floor(((shape_m * shape_n) / 100) * cells_to_fill)
#     source_format = formats.get(source_format)
#     timer = None
#     while True:
#         target_format = yield timer
#         if converter := formats.get(target_format):
#             timer = Timer(
#                 setup=setup % (
#                     source_format, converter, source_format, shape_m,
#                     cells_to_fill, data_type
#                 ),
#                 stmt=stmt % (converter,),
#             )
#         else:
#             timer = None


def collector(
        converter: Generator[Timer, str, None],
        formats: List[str],
        repeat: int,
        number: int):
    all_timings = {}
    for target_format in formats:
        timer = converter.send(target_format)
        timings = timer.repeat(repeat=repeat, number=number)
        all_timings[target_format] = asarray(timings)
    return all_timings


def main():
    repeat = 10
    number = 1_000
    all_formats = ["bsr", "coo", "csc", "csr"]
    # matrix_shape = [100_000, 100_000]
    # nonzero_cells = floor(mul(*matrix_shape) / 100 * 0.01)
    path = Path(r"C:\Users\Maxximiser\Downloads\tm_reincarnation.npz")
    matrix = scipy_sparse.load_npz(path)
    # nonzero_cells = 1000000

    metadata = ConversionMetadata(
        title="SciPy sparse format conversions",
        run_date=datetime.now().isoformat(),
        number_of_repeats=repeat,
        number_of_runs=number,
        conversions=all_formats,
        matrix_shape=matrix.shape,
        nonzero_cells=matrix.count_nonzero()
    )

    out_path = Path.cwd() / Path("scipy_reinc")
    if not out_path.exists():
        out_path.mkdir()

    with (out_path / "metadata.json").open("w", encoding="utf-8") as file:
        file.write(dumps(asdict(metadata), indent=4))

    for source_format in all_formats:
        scipy_gen = scipy_converter(source_format, path)
        next(scipy_gen)
        results = DataFrame(
            collector(scipy_gen, all_formats, repeat=repeat, number=number)
        )
        results.to_parquet(out_path / f"{source_format}.parquet")
        scipy_gen.close()

    # TODO: Convert input matrix to right format!


if __name__ == "__main__":
    main()
