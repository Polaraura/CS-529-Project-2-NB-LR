import dask.array as da
import numpy as np


def normalize_column_vector(column_vector):
    # vector_sum = da.sum(column_vector).compute()
    vector_sum = np.sum(column_vector)

    assert vector_sum != 0.0

    # print(f"vector: {column_vector[0:6]}")
    # print(f"sum: {vector_sum}")
    # print(f"new vector: {(column_vector / vector_sum)[0:6]}")

    return column_vector / vector_sum