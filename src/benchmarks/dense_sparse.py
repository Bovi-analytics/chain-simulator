from timeit import timeit, Timer
import seaborn as sns

import matplotlib.pyplot as plt


def multiply_dense():
    timer = timeit(
        stmt="""array.dot(array)""",
        setup="""import numpy as np; array = np.random.rand(100, 100)""",
        number=10_000
    )
    print(timer)


def multiply_sparse_csr():
    timer = timeit(
        stmt="array.dot(array)",
        setup="import scipy; import numpy as np; array = scipy.sparse.random(100, 100, density=1, random_state=1, format='csr', dtype=int)",
        number=10_000
    )
    print(timer)


def multiply_sparse_csc():
    timer = timeit(
        stmt="array.dot(array)",
        setup="import scipy; import numpy as np; array = scipy.sparse.random(100, 100, density=1, random_state=1, format='csc', dtype=int)",
        number=10_000
    )
    print(timer)


def dense_sparse():
    shape = "1_000, 1_000"
    densities = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
    stmt = "array.dot(array)"
    x = []
    y = []
    for density in reversed(densities):
        setup = f"import scipy; array = scipy.sparse.random({shape}, density={density}, random_state=1, format='csr')"
        timer = timeit(stmt, setup, number=1_000)
        x.append(density)
        y.append(timer)
        # data.append((timer, density))
        print(f"Took {timer} with density {density}")
    plt.scatter(x, y)
    plt.show()
    # sns.scatterplot(data, x="Time (s)", y="Density").imshow()


def dense_scale():
    # shapes = (1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000)
    # shapes = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    shapes = range(1024, 4096, 8)
    stmt = "array.dot(array)"
    for shape in shapes:
        setup = f"import numpy as np; array = np.random.rand({shape}, {shape})"
        timer = Timer(stmt, setup)
        # iterations = timer.autorange()
        iterations = 20
        results = timer.repeat(number=iterations, repeat=10)
        print(f"Shape {shape}, best of 10 in {iterations} iterations: {min(results)}, {min(results) / iterations}")



if __name__ == "__main__":
    multiply_dense()
    # multiply_dense()
    # multiply_sparse_csr()
    # multiply_sparse_csc()
    # dense_sparse()
    dense_scale()
