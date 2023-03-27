import dask.array as da
import dask.dataframe as df
import numpy as np
import pandas

from utilities.DebugFlags import LOGISTIC_REGRESSION_PREDICTION_DEBUG


def normalize_column_vector(column_vector):
    """
    Using np.sum() is much faster than da.sum()...and the column vector is passed in as a numpy array

    :param column_vector:
    :return:
    """

    # if LOGISTIC_REGRESSION_PREDICTION_DEBUG:
    #     print(f"normalizing column vector...")

    # need to take the abs before taking the sum
    # vector_sum = da.sum(column_vector).compute()
    vector_sum = np.sum(np.abs(column_vector))

    # if the column is all 0s, return the vector unmodified
    if vector_sum == 0.0:
        return column_vector

    # print(f"vector: {column_vector[0:6]}")
    # print(f"sum: {vector_sum}")
    # print(f"new vector: {(column_vector / vector_sum)[0:6]}")

    return column_vector / vector_sum


def calculate_entropy_column_vector(p_vector):
    return -np.sum(p_vector * np.log2(p_vector))


def calculate_gini_index_column_vector(p_vector):
    return 1 - np.sum(np.power(p_vector, 2))


def get_class_count_column_vector(count_vector):
    pass


if __name__ == "__main__":
    a = np.array([[1, 2, 3], [2, 3, 4]])

    print(np.apply_along_axis(calculate_entropy_column_vector, 0, a))

    b = da.from_array([0, 1, 2, 3, 4, 5, 0], chunks=(1,))
    print(f"{b.map_blocks(lambda x: da.ones((8, )).compute()).compute()}")
    print(f"{da.apply_along_axis(lambda x: np.ones((8, )), 0, b).compute()}")

    print(f"{(b == 0).compute()}")
    c = da.from_array([True, True, True, False, False, False, False])

    d = da.where(da.logical_and(b == 0, c == True), True, False)
    print(f"{d.compute()}")

    # e = df.from_dask_array(da.zeros((12000, 60000)), columns=list(range(60000)))
    # print(f"{e[0].value_counts().compute()}")

    f = df.from_dask_array(da.from_array([[1, 2], [3, 4]]), columns=[1, 2])
    print(f)
    print(f.compute())
    print(f[1])
    print(f[1].compute())
    print(f"{f[1].value_counts().compute()}")
    print(f"{f[1].value_counts()[1].compute()}")
    print(f"{f[1].value_counts().index.compute()}")
    print(f"{f[1].value_counts().index.compute().to_list()}")
    # print(f"{da.from_array(f[1].value_counts().index.compute())[0].compute()}")
    print(f"{f[1].value_counts().values.compute()}")

    b_index = f[1].value_counts().index.compute()
    b_index += 1
    b_values = f[1].value_counts().values.compute()
    b_values[0] = 0

    print(b.compute())
    b[b_index] = b_values
    print(b.compute())

    g = df.from_dask_array(b, columns="a")
    g_panda = pandas.DataFrame(b, columns=["a"])
    print(g)
    print(g.compute())

    # print(g[da.logical_and(b == 0, c == True)].value_counts().compute())
    # print(g[g == 0 or g == 2].value_counts().compute())

    # print(g_panda[g_panda == 0 & g_panda == 0].value_counts())



