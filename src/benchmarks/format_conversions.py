from dataclasses import dataclass
from json import dumps
from pathlib import Path
from textwrap import dedent
from timeit import Timer
from typing import Generator, List

from cupyx.profiler import benchmark
from cupyx.profiler._time import _PerfCaseResult
from cupyx.scipy import sparse as cupy_sparse
from numpy import asarray
from pandas import DataFrame
from scipy import sparse as scipy_sparse


@dataclass
class ConversionMetadata:
    title: str
    number_of_repeats: int
    number_of_warmups: int
    conversions: List[str]
    matrix_shape: List[int]
    nonzero_cells: int


def scipy_converter(
        npz_file: Path,
        source_format: str,
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
    setup = dedent("""
        from scipy.sparse import load_npz, %s, %s

        initial_array = %s(load_npz(r'%s'))
    """).strip()
    stmt = "converted_array = %s(initial_array)"
    source_format = formats.get(source_format)
    timer = None
    while True:
        target_format = yield timer
        print(f"To {target_format}")
        if converter := formats.get(target_format):
            timer = Timer(
                setup=setup % (
                    source_format, converter, source_format, npz_file,
                ),
                stmt=stmt % (converter,),
            )
        else:
            timer = None


def cupy_converter(
        source_format: str,
        matrix,
        n_warmup: int,
        n_repeats: int,
) -> Generator[_PerfCaseResult, str, None]:
    formats = {
        "coo": cupy_sparse.coo_matrix,
        "csc": cupy_sparse.csc_matrix,
        "csr": cupy_sparse.csr_matrix,
        # "dia": cupy_sparse.dia_matrix,
    }
    source_format = formats.get(source_format)
    source = source_format(matrix)
    timer = None
    while True:
        target_format = yield timer
        if converter := formats.get(target_format):
            timer = benchmark(converter, (source.copy(),), n_warmup=n_warmup, n_repeat=n_repeats)
        else:
            timer = None


def scipy_collector(
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


def cupy_collector(
        converter: Generator[_PerfCaseResult, str, None],
        formats: List[str],
):
    cpu_timings = {}
    gpu_timings = {}
    for target_format in formats:
        timer = converter.send(target_format)
        # timings = {
        #     "cpu_times": timer.cpu_times,
        #     "gpu_times": timer.gpu_times
        # }
        cpu_timings[target_format] = timer.cpu_times
        gpu_timings[target_format] = timer.gpu_times[0]
    return DataFrame(cpu_timings), DataFrame(gpu_timings)


def benchmark_scipy(npz_file: Path, out_path: Path, number: int, repeat: int, all_formats):
    metadata = {
        "number_of_repeats": repeat,
        "number_of_runs": number,
    }

    out_path = out_path / Path(npz_file.stem) / Path("scipy_timings")
    if not out_path.exists():
        out_path.mkdir(parents=True)

    with (out_path / "metadata.json").open("w", encoding="utf-8") as file:
        file.write(dumps(metadata, indent=4))

    for source_format in all_formats:
        print(f"Converting from {source_format}")
        scipy_gen = scipy_converter(npz_file, source_format)
        next(scipy_gen)
        timings = DataFrame(scipy_collector(scipy_gen, all_formats, repeat, number))
        timings.to_parquet(out_path / f"{source_format}.parquet")
        scipy_gen.close()


def benchmark_cupy(npz_file: Path, out_path: Path, warmup: int, repeat: int, all_formats):
    matrix = scipy_sparse.load_npz(npz_file)
    cupy_matrix = cupy_sparse.coo_matrix(matrix)

    metadata = {
        "number_of_repeats": repeat,
        "number_of_warmups": warmup,
    }

    out_path = out_path / Path(npz_file.stem) / Path("cupy_timings")
    if not out_path.exists():
        out_path.mkdir(parents=True)

    with (out_path / "metadata.json").open("w", encoding="utf-8") as file:
        file.write(dumps(metadata, indent=4))

    for source_format in all_formats:
        cupy_gen = cupy_converter(source_format, cupy_matrix, n_warmup=warmup, n_repeats=repeat)
        next(cupy_gen)
        cpu_timings, gpu_timings = cupy_collector(cupy_gen, all_formats)
        cpu_timings.to_parquet(out_path / f"{source_format}_cpu.parquet")
        gpu_timings.to_parquet(out_path / f"{source_format}_gpu.parquet")
        cupy_gen.close()


if __name__ == "__main__":
    base_input_path = Path("docs/source/benchmarks/data")
    base_output_path = base_input_path / "format_conversions"
    path_matrices = base_input_path / "Matrices"

    # Define paths to all matrices to test.
    random_tiny = path_matrices / "random_matrix_tiny.npz"
    random_small = path_matrices / "random_matrix_small.npz"
    random_medium = path_matrices / "random_matrix_medium.npz"
    random_large = path_matrices / "random_matrix_large.npz"
    with_reincarnation = path_matrices / "tm_with_reincarnation.npz"
    without_reincarnation = path_matrices / "tm_without_reincarnation.npz"

    # Run SciPy benchmarks
    # benchmark_scipy(
    #     random_tiny, base_output_path, 2_000, 10,
    #     ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]
    # )
    # benchmark_scipy(
    #     random_small, base_output_path, 500, 10,
    #     ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]
    # )

    # benchmark_scipy(
    #     random_medium, base_output_path, 1_000, 10,
    #     ["bsr", "coo", "csc", "csr"]
    # )  # dia format requires 54.7 GiB of memory
    # benchmark_scipy(
    #     random_large, base_output_path, 200, 10,
    #     ["bsr", "coo", "csc", "csr"]
    # )
    # benchmark_scipy(
    #     with_reincarnation, base_output_path, 100, 10,
    #     ["bsr", "coo", "csc", "csr"]
    # )
    # benchmark_scipy(
    #     with_reincarnation, base_output_path, 100, 10,
    #     ["bsr", "coo", "csc", "csr"]
    # )

    # Run CuPy benchmarks
    # benchmark_cupy(
    #     random_tiny, base_output_path, 10, 10_000, ["coo", "csc", "csr"]
    # )
    benchmark_cupy(
        random_small, base_output_path, 10, 2_500, ["coo", "csc", "csr"]
    )


    # benchmark_cupy(
    #     random_small, base_output_path, 10, 10_000, ["coo", "csc", "csr"]
    # )

    # f = Path(
    #     r"C:\Users\Maxximiser\Documents\Software projects\Python projects\Intern Utrecht\simulation_platform\tm_reincarnation.npz")
    # o = Path(
    #     r"C:\Users\Maxximiser\Documents\Software projects\Python projects\Intern Utrecht\simulation_platform\conversion_times")
    # benchmark_scipy(f, o, 200, 10, ["bsr", "coo", "csc", "csr"])
    # benchmark_cupy(f, o, 10, 10_000, ["coo", "csc", "csr"])
