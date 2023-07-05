from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, asdict
from json import dumps
from pathlib import Path
from textwrap import dedent
from time import perf_counter
from timeit import Timer

from cupy import zeros as cupy_zeros
from cupyx.profiler import benchmark
from cupyx.scipy import sparse as cupy_sparse
from numpy import asarray
from pandas import DataFrame
from scipy import sparse as scipy_sparse


@dataclass
class BenchmarkMetadata(ABC):
    duration_seconds: float
    number: int
    repeat: int


@dataclass
class CuPyBenchmarkMetadata(BenchmarkMetadata):
    warmup: int


@dataclass
class SciPyBenchmarkMetadata(BenchmarkMetadata):
    function_iterations: int


def write_results(
        device: str,
        number: int,
        source_format: str,
        metadata: BenchmarkMetadata,
        timings: DataFrame,
        out_path: Path):
    operation = "multiply_once" if number == 1 else "multiply_iteratively"
    filename_base = f"{device}_{source_format}_{operation}"
    filename_metadata = f"{filename_base}_metadata.json"
    filename_timings = f"{filename_base}.parquet"
    if not out_path.exists():
        out_path.mkdir(parents=True)
    with (out_path / filename_metadata).open("w", encoding="utf-8") as file:
        file.writelines(dumps(asdict(metadata), indent=4))
    timings.to_parquet(out_path / filename_timings)


def cupy_vector_matrix(dot_function, left, right, n_times):
    if n_times > 1:
        result = left
        for _ in range(n_times):
            result = dot_function(result, right)
    else:
        result = dot_function(left, right)


def benchmark_cupy(npz_file: Path | str, source_format: str, warmup, number,
                   repeat, out_path: Path, method: str = "vector_matrix"):
    formats = {
        "coo": cupy_sparse.coo_matrix,
        "csc": cupy_sparse.csc_matrix,
        "csr": cupy_sparse.csr_matrix,
    }
    if fmt := formats.get(source_format):
        right = fmt(scipy_sparse.load_npz(npz_file))
        if method == "vector_matrix":
            left = cupy_zeros(right.get_shape()[0])
            left[0] = 1.0
        elif method == "matrix_matrix":
            left = right
        else:
            raise ValueError(f"Unknown method `{method}`")
        time_start = perf_counter()
        timings = benchmark(cupy_vector_matrix,
                            (fmt.dot, left, right, number), n_warmup=warmup,
                            n_repeat=repeat)
        time_end = perf_counter()
        metadata = CuPyBenchmarkMetadata(
            duration_seconds=time_end - time_start,
            number=number, repeat=repeat, warmup=warmup
        )
        df = DataFrame(
            {
                "device": "gpu",
                "operation_method": "once" if number == 1 else "iterative",
                "sparse_format": source_format,
                "duration_cpu": timings.cpu_times,
                "duration_gpu": timings.gpu_times[0]
            }
        )
        write_results("gpu", number, source_format, metadata, df, out_path)
    else:
        raise ValueError(f"Unknown format `{source_format}`")


def benchmark_scipy(npz_file: Path | str, source_format: str, iterations,
                    number, repeat, out_path, method: str = "vector_matrix"):
    formats = {
        "bsr": "bsr_array",
        "csc": "csc_array",
        "csr": "csr_array",
    }
    if fmt := formats.get(source_format):
        setup = dedent("""
            from numpy import zeros
            from scipy.sparse import load_npz, %s

            right = %s(load_npz(r'%s'))
            if "%s" == "vector_matrix":
                left = zeros(right.get_shape()[0])
                left[0] = 1.0
            elif "%s" == "matrix_matrix":
                left = right
            else:
                raise ValueError(f"Unknown method `{'%s'}`")
        """).strip()
        stmt = dedent("""
                if %d > 1:
                    result = left
                    for _ in range(%d):
                        result = %s.dot(result, right)
                else:
                    result = %s.dot(left, right)
        """).strip()
        timer = Timer(
            setup=setup % (fmt, fmt, npz_file, method, method, method),
            stmt=stmt % (iterations, iterations, fmt, fmt)
        )
        time_start = perf_counter()
        timings = timer.repeat(number=number, repeat=repeat)
        time_end = perf_counter()
        metadata = SciPyBenchmarkMetadata(
            duration_seconds=time_end - time_start,
            number=number, repeat=repeat, function_iterations=iterations
        )
        df = DataFrame(
            {
                "device": "cpu",
                "operation_method": "once" if number == 1 else "iterative",
                "sparse_format": source_format,
                "duration": asarray(timings)
            }
        )
        write_results("cpu", number, source_format, metadata, df, out_path)
    else:
        raise ValueError(f"Unknown format `{source_format}`")


def main(input_file: Path, output_path: Path):
    out_path_vector = output_path / "arithmetic_vector_matrix" / input_file.stem
    out_path_matrix = output_path / "arithmetic_matrix_matrix" / input_file.stem
    for scipy_format in ["bsr", "csc", "csr"]:
        # benchmark_scipy(input_file, scipy_format, 1, 1_000,
        #                 10, out_path_vector)
        # benchmark_scipy(input_file, scipy_format, 1_000, 1,
        #                 10, out_path_vector)
        benchmark_scipy(input_file, scipy_format, 32, 1,
                        10, out_path_matrix, method="matrix_matrix")
    # for cupy_format in ["coo", "csc", "csr"]:
        # benchmark_cupy(input_file, cupy_format, 20, 1,
        #                10_000, out_path_vector)
        # benchmark_cupy(input_file, cupy_format, 20, 1_000,
        #                10, out_path_vector)
        # benchmark_cupy(input_file, cupy_format, 20, 32,
        #                10, out_path_matrix, method="matrix_matrix")


if __name__ == "__main__":
    in_files = [
        Path("docs/source/benchmarks/data/matrices/transition_matrix_acyclic.npz"),
        Path("docs/source/benchmarks/data/matrices/transition_matrix_cyclic.npz"),
    ]
    out_directory = Path("docs/source/benchmarks/data")
    for in_file in in_files:
        main(in_file, out_directory)