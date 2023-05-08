from scipy.sparse import coo_array


def chain_simulator(array: coo_array, steps: int) -> coo_array:
    """Progress a Markov chain forward in time.

    Method which progresses a Markov chain forward in time using a provided
    transition matrix. Based on the `steps` parameter, the transition matrix is
    raised to the power of `steps`. This is done using a matrix multiplication.

    :param array: Transition matrix.
    :type array: coo_array
    :param steps: Steps in time to progress the simulation.
    :type steps: int
    :return: Transition matrix progressed in time.
    :rtype coo_array
    """
    compressed_array = array.tocsr()
    if steps == 1:
        return compressed_array @ compressed_array
    progressed_array = compressed_array
    for _step in range(steps):
        progressed_array = compressed_array @ progressed_array
    return progressed_array.tocoo()
