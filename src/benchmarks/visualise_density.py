import click
import scipy
from dataclasses import dataclass
from pathlib import Path
from chain_simulator._utilities import _plot_coo_matrix
import matplotlib.pyplot as plt


@click.option("-i", "--input", multiple=True)
@click.option("-o", "--output")
def main(*input: str, output: str):
    for file in input:
        matrix = scipy.sparse.load_npz(file)


def vis():
    input_files = [
        Path(r"C:\Users\Maxximiser\Downloads\tm_no_reincarnation.npz"),
        # Path(r"C:\Users\Maxximiser\Downloads\tm_reincarnation.npz")
    ]
    for input_file in input_files:
        out_dir = Path.joinpath(input_file.parent, input_file.stem)
        out_dir.mkdir() if not out_dir.exists() else None
        mx = scipy.sparse.load_npz(input_file)
        ax, fig = _plot_coo_matrix(mx)
        name = f"n{1}.jpg"
        ax.figure.savefig(out_dir / name)
        plt.close()
        print(mx.shape)
        print(mx.count_nonzero())

        n1 = mx.tocsr()
        n2 = n1.dot(n1)
        n4 = n2.dot(n2)
        n8 = n4.dot(n4)
        n16 = n8.dot(n8)

        ax, fix = _plot_coo_matrix(n16.tocoo())
        ax.figure.savefig(out_dir / "n16.jpg")
        plt.close()
        print(n16.count_nonzero())
        plot_increment(n16, 16, 256, 16, out_dir)


def plot_increment(transition_matrix: scipy.sparse.csr_array, start: int, end: int, step: int, path: Path):
    progressed = transition_matrix.copy()
    for index in range(start + step, end + 1, step):
        print(index)
        progressed = progressed.dot(transition_matrix)
        ax, fig = _plot_coo_matrix(progressed.tocoo())
        ax.figure.savefig(path / f"n{index}.jpg")
        plt.close()
        print(progressed.count_nonzero())


if __name__ == "__main__":
    # main()
    vis()

