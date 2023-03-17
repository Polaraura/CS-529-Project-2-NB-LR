import dask.array as da
import numpy as np


def normalize_column_vector(column_vector):
    """
    Using np.sum() is much faster than da.sum()...and the column vector is passed in as a numpy array

    FIXME: need to take the abs BEFORE getting the sum...

    :param column_vector:
    :return:
    """

    # vector_sum = da.sum(column_vector).compute()
    vector_sum = np.sum(np.abs(column_vector))

    # if the column is all 0s, return the vector unmodified
    if vector_sum == 0.0:
        return column_vector

    # print(f"vector: {column_vector[0:6]}")
    # print(f"sum: {vector_sum}")
    # print(f"new vector: {(column_vector / vector_sum)[0:6]}")

    return column_vector / vector_sum
