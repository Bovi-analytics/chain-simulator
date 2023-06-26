from pathlib import Path
from scipy import sparse as scipy_sparse
from pandas import DataFrame


def main(input_path: Path, output_path: Path):
    information = {
        "name": [],
        "axis_size": [],
        "total_cells": [],
        "nonzero_count": [],
        "density": [],
        "size_as_dense": [],
    }

    for matrix_npz_file in input_path.glob("*.npz"):
        matrix = scipy_sparse.load_npz(matrix_npz_file)
        information["name"].append(matrix_npz_file.stem)
        information["axis_size"].append(matrix.shape[0])
        information["total_cells"].append(matrix.shape[0] * matrix.shape[1])
        information["nonzero_count"].append(matrix.count_nonzero())
        information["density"].append(
            (100 / information["total_cells"][-1]) * information["nonzero_count"][-1]
        )
        information["size_as_dense"].append(information["total_cells"][-1] * matrix.data.itemsize)
    output_path = output_path / "basic_matrix_info"
    if not output_path.exists():
        output_path.mkdir(parents=True)
    DataFrame(information).to_parquet(output_path / "basic_matrix_info.parquet")


if __name__ == "__main__":
    main(
        Path("docs/source/benchmarks/data/matrices"),
        Path("docs/source/benchmarks/data")
    )
