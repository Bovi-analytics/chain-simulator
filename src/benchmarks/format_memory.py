import warnings
from pathlib import Path

from numpy.core._exceptions import _ArrayMemoryError
from pandas import DataFrame
from pympler.asizeof import asizeof
from scipy import sparse as scipy_sparse


def main(path_in: Path, path_out: Path, *target_formats):
    data = {
        "name": [],
        "sparse_format": [],
        "size_bytes": [],
    }
    for npz_file in path_in.glob("*.npz"):
        matrix = scipy_sparse.load_npz(npz_file)
        analyser = scipy_memory_footprint(matrix.copy())
        next(analyser)
        for target_format in target_formats:
            data["name"].append(npz_file.stem)
            data["sparse_format"].append(target_format)
            data["size_bytes"].append(analyser.send(target_format))
        analyser.close()
    dataframe = DataFrame(data)
    if not path_out.exists():
        path_out.mkdir(parents=True)
    dataframe.to_parquet(path_out / "memory_footprints.parquet")


def scipy_memory_footprint(matrix):
    formats = {
        "bsr": scipy_sparse.bsr_array,
        "coo": scipy_sparse.coo_array,
        "csc": scipy_sparse.csc_array,
        "csr": scipy_sparse.csr_array,
        "dia": scipy_sparse.dia_array,
        "dok": scipy_sparse.dok_array,
        "lil": scipy_sparse.lil_array,
    }
    matrix_size = None
    while True:
        target_format = yield matrix_size
        if converter := formats.get(target_format):
            try:
                converted = converter(matrix.copy())
            except _ArrayMemoryError as exc:
                warnings.warn(str(exc))
                converted = None
            matrix_size = asizeof(converted) if converted is not None else None
        else:
            raise ValueError(f"Unknown sparse format `{target_format}`!")


if __name__ == "__main__":
    all_formats = ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]
    input_path = Path("docs/source/benchmarks/data/matrices")
    output_path = input_path.parent / "memory_footprints"
    main(input_path, output_path, *all_formats)
