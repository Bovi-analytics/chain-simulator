from pathlib import Path

import click
import matplotlib.pyplot as plt
import scipy
from scipy import sparse as scipy_sparse

from chain_simulator._utilities import _plot_coo_matrix


@click.option("-i", "--input", multiple=True)
@click.option("-o", "--output")
def main(*input: str, output: str):
    for file in input:
        matrix = scipy.sparse.load_npz(file)


def vis(npz_file: Path, output_path: Path, n: int):
    # f, ax = plt.subplots(figsize=(5.5, 4.5))
    output_path = output_path / npz_file.stem
    if not output_path.exists():
        output_path.mkdir(parents=True)
    name = npz_file.stem.capitalize().replace("_", " ")

    # Plot matrix as-is.
    matrix = scipy_sparse.load_npz(npz_file).tocoo()
    ax, fig = _plot_coo_matrix(matrix, name, 1)
    plt.tight_layout()
    plt.savefig(output_path / "n1.jpg", dpi=350)
    plt.close()

    # Progress if n >= 32
    if n < 32:
        return
    n1 = matrix.tocsr()
    n2 = n1.dot(n1)
    n4 = n2.dot(n2)
    n8 = n4.dot(n4)
    n16 = n8.dot(n8)
    n32 = n16.dot(n16)
    ax, fix = _plot_coo_matrix(n32.tocoo(), name, 32)
    plt.tight_layout()
    plt.savefig(output_path / "n32.jpg", dpi=350)
    plt.close()

    # Progress if n >= 64
    if n < 64:
        return
    plot_increment(n32, 32, 256, 32, output_path, name)

    # input_files = [
    #     Path(r"C:\Users\Maxximiser\Downloads\tm_no_reincarnation.npz"),
    #     # Path(r"C:\Users\Maxximiser\Downloads\tm_reincarnation.npz")
    # ]
    # for input_file in input_files:
    #     out_dir = Path.joinpath(input_file.parent, input_file.stem)
    #     out_dir.mkdir() if not out_dir.exists() else None
    #     mx = scipy.sparse.load_npz(input_file)
    #     ax, fig = _plot_coo_matrix(mx)
    #     name = f"n{1}.jpg"
    #     ax.figure.savefig(out_dir / name)
    #     plt.close()
    #     print(mx.shape)
    #     print(mx.count_nonzero())
    #
    #     n1 = mx.tocsr()
    #     n2 = n1.dot(n1)
    #     n4 = n2.dot(n2)
    #     n8 = n4.dot(n4)
    #     n16 = n8.dot(n8)
    #
    #     ax, fix = _plot_coo_matrix(n16.tocoo())
    #     ax.figure.savefig(out_dir / "n16.jpg")
    #     plt.close()
    #     print(n16.count_nonzero())
    #     plot_increment(n16, 16, 256, 16, out_dir)


def plot_increment(transition_matrix: scipy.sparse.csr_array, start: int,
                   end: int, step: int, path: Path, name: str):
    progressed = transition_matrix.copy()
    for index in range(start + step, end + 1, step):
        progressed = progressed.dot(transition_matrix)
        ax, fig = _plot_coo_matrix(progressed.tocoo(), name, index)
        plt.tight_layout()
        ax.figure.savefig(path / f"n{index}.jpg", dpi=350)
        plt.close()
        # print(progressed.count_nonzero())


if __name__ == "__main__":
    files = Path("docs/source/benchmarks/data/matrices").glob("*.npz")
    out_path = Path("docs/source/benchmarks/data/matrix_visualisations")
    for file in files:
        total_steps = 256 if "cow" in file.stem else 32
        if "cow" not in file.stem:
            vis(file, out_path, total_steps)
        # if "cow" in file.stem:
        #     total_steps = 256
        # else:
        #     total_steps = 32

