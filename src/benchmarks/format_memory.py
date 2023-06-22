from pympler.asizeof import asizeof
from scipy import sparse as scipy_sparse
from pandas import Series


def main(matrix):
    conversions = ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]
    tool = scipy_memory(matrix.copy())
    next(tool)
    sizes = {fmt: tool.send(fmt) for fmt in conversions}
    tool.close()
    df = Series(sizes).to_frame().transpose()
    df.to_parquet("memfootprint.parquet")
    print(df)


def scipy_memory(matrix):
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
            matrix_size = asizeof(converter(matrix.copy()))
        else:
            raise ValueError(f"Unknown sparse format `{target_format}`!")


if __name__ == "__main__":
    mx = scipy_sparse.load_npz(r"C:\Users\Maxximiser\Downloads\tm_reincarnation.npz")
    main(mx)
