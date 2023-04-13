from textwrap import dedent
from timeit import Timer


def iterate_lil():
    setup = dedent("""
    from numpy import array
    from numpy.random import default_rng
    from math import pow
    from scipy.sparse import lil_array

    axis_size = 1_024
    axis_range = array(range(0, axis_size))
    cells_count = int(pow(axis_size, 2) * 0.5)
    rows = default_rng(1).choice(axis_range, cells_count)
    cols = default_rng(1).choice(axis_range, cells_count)
    data = default_rng(1).random(cells_count)
    """)
    stmt = dedent("""
    array = lil_array((axis_size, axis_size), dtype="float64")
    for index_row, index_col, num in zip(rows, cols, data):
        array[index_row, index_col] = num
    """)
    timer = Timer(stmt, setup)
    data = timer.repeat(10, 10)
    print(data)


def construct_csr():
    setup = dedent("""
        from numpy import array
        from numpy.random import default_rng
        from math import pow
        from scipy.sparse import csr_array

        axis_size = 3_072
        axis_range = array(range(0, axis_size))
        cells_count = int(pow(axis_size, 2) * 0.5)
        rows = default_rng(1).choice(axis_range, cells_count)
        cols = default_rng(1).choice(axis_range, cells_count)
        data = default_rng(1).random(cells_count)
        """)
    stmt = dedent("""
        csr_array((data, (rows, cols)), shape=(axis_size, axis_size))
        """)
    timer = Timer(stmt, setup)
    data = timer.repeat(10, 10)
    print(data)


def construct_dok():
    setup = dedent("""
        from numpy import array
        from numpy.random import default_rng
        from math import pow
        from scipy.sparse import coo_array

        axis_size = 3_072
        axis_range = array(range(0, axis_size))
        cells_count = int(pow(axis_size, 2) * 0.5)
        rows = default_rng(1).choice(axis_range, cells_count)
        cols = default_rng(1).choice(axis_range, cells_count)
        data = default_rng(1).random(cells_count)
        """)
    stmt = dedent("""
        coo_array((data, (rows, cols)), shape=(axis_size, axis_size))
    """)
    timer = Timer(stmt, setup)
    data = timer.repeat(10, 10)
    print(data)


if __name__ == "__main__":
    # iterate_lil()
    construct_csr()
    construct_dok()
