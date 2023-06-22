from __future__ import annotations

from numpy import asarray
from scipy import sparse as scipy_sparse
from numpy.random import default_rng
from pathlib import Path
import click


@click.command()
@click.option("--axis_size", type=int, help="Size of one axis")
@click.option("--cells_to_fill", type=int, help="Number of cells to fill")
@click.option("--out_path", type=Path, help="Output path to write matrix to")
@click.option("--data_type", type=str, default="float64", help="Optional, datatype of nonzero values", required=False)
def main(
    axis_size: int, cells_to_fill: int, data_type: str, out_path: Path | str
):
    array = build_array(axis_size, cells_to_fill, data_type)
    if isinstance(out_path, str):
        path = Path(out_path)
    elif isinstance(out_path, Path):
        path = out_path
    else:
        raise TypeError(f"Unknown type for out_path: `{type(out_path)}`")
    scipy_sparse.save_npz(path, array)


def build_array(axis_size: int, cells_to_fill: int, data_type: str):
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


if __name__ == "__main__":
    main()