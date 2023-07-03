from pathlib import Path

from pandas import DataFrame
from scipy import sparse as scipy_sparse
from math import log2
from operator import mul


def main(output_file: Path, *npz_files: Path):
    data = {
        "matrix_name": [],
        "step_in_time": [],
        "total_cells": [],
        "nonzero_cells": [],
    }
    for npz_file in npz_files:
        matrix = scipy_sparse.load_npz(npz_file)
        progressor = progress_matrix(matrix.tocsr(), 1, 256, 8)
        for progressed, step_in_time in progressor:
            data["matrix_name"].append(npz_file.stem)
            data["step_in_time"].append(step_in_time)
            data["total_cells"].append(mul(*progressed.shape))
            data["nonzero_cells"].append(progressed.count_nonzero())
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    DataFrame(data).to_parquet(output_file)


def progress_matrix(matrix, step_initial, step_target, step_size):
    yield matrix, 1
    if step_initial == 1 and step_size != 1:
        steps_intermediate = log2(step_size)
        if steps_intermediate != int(steps_intermediate):
            raise ValueError(f"Step size `{step_size}` has no integer log2 result")
        matrix_step = None
        tally = 1
        for step_intermediate in range(1, int(steps_intermediate) + 1):
            tally *= 2
            if matrix_step is None:
                matrix_step = matrix.dot(matrix)
            else:
                matrix_step = matrix_step.dot(matrix_step)
        matrix_progressed = matrix_step
        for index in range(tally, step_target + 1, step_size):
            if index == step_size:
                yield matrix_progressed, index
            else:
                matrix_progressed = matrix_progressed.dot(matrix_step)
                yield matrix_progressed, index
    else:
        raise ValueError


if __name__ == "__main__":
    all_npz_files = Path("docs/source/benchmarks/data/matrices").glob("*cow*.npz")
    out = Path("docs/source/benchmarks/data/density_change/density_change.parquet")
    main(out, *all_npz_files)
