import time
from io import StringIO
import os

import numpy as np
import pandas as pd
from pandas.arrays import SparseArray
import scipy
from numpy import float64

import dask.dataframe as ddf
import dask.bag as db
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

import pickle

import sparse

from utilities.Constants import CHUNK_SIZE


def parse_class_labels(input_filepath: str):
    class_labels_dict = {}

    with open(input_filepath, 'r') as input_file:
        for line in input_file:
            key, val = map(str.strip, line.split(' '))
            key = int(key)
            class_labels_dict[key] = val

    return class_labels_dict


def parse_data_training_array(input_filepath: str, output_filepath: str, chunksize=CHUNK_SIZE):
    # data = ddf.read_csv(filename, header=None)

    # only reads line by line...
    print(f"initial read text")
    data = db.read_text(input_filepath, blocksize="10MB", linedelimiter="\n")
    # for d in data:
    #     print(f"{d}")

    print(f"map to numpy array")
    # data = data.map(lambda x: np.loadtxt(StringIO(x), delimiter=','))
    data = data.map(lambda x: np.fromstring(x, dtype=int, sep=','))
    # print(f"{data.compute()}")

    # data_array = data.to_dask_array(lengths=True)
    # data_array = da.concatenate(data, axis=0)
    print(f"map to dask array")
    data = data.map(lambda x: da.from_array(x))
    print(f"{data}")

    print(f"map to sparse dask array")
    data_array = data.map(lambda x: x.map_blocks(lambda x: sparse.COO(x, fill_value=0)))

    print(f"{data_array}")

    sparse_chunks = data_array

    ################################################

    print(f"stack sparse dask array")
    sparse_chunks = da.stack(sparse_chunks, axis=0)
    #sparse_chunks = sparse_chunks.compute_chunk_sizes()

    print(f"{sparse_chunks}")
    print(f"{sparse_chunks.compute()}")

    # with open(output_filepath, 'wb') as output_file:
    #     # FIXME: can't pickle the lazy object...
    #     # pickle.dump(sparse_chunks, output_file)
    #     pickle.dump(sparse_chunks.compute(), output_file)

    save_da_array_pickle(sparse_chunks, output_filepath)

    return sparse_chunks

    ################################################


def save_da_array_pickle(array: da.array, output_filepath: str):
    """
    Saves the dask Array object into a pickle file (need to call compute() first since the array is just a delayed
    object with no values before the call to compute())

    :param array:
    :param output_filepath:
    :return:
    """

    with open(output_filepath, 'wb') as output_file:
        # FIXME: can't pickle the lazy object...
        # pickle.dump(sparse_chunks, output_file)
        pickle.dump(array.compute(), output_file)


def load_da_array_pickle(output_filepath: str):
    """
    computed object...

    IMPORTANT: chunk size is set to auto (default optional arg)
    initially read in row by row with chunk size (1, 61190) -- pretty slow
    auto set chunk size to (5792, 5792) -- much faster (AFTER .pkl file was created)
    see test computation below for timings comparison

    :param output_filepath:
    :return:
    """

    with open(output_filepath, 'rb') as output_file:
        sparse_da_training = pickle.load(output_file)
        sparse_da_training = da.from_array(sparse_da_training)
        # chunks=(12000, 1)

    return sparse_da_training


if __name__ == "__main__":
    pbar = ProgressBar()
    pbar.register()  # global registration

    # introduce parallelism with multiple workers/threads...slower though and can't see the progress bar
    # local_cluster = LocalCluster()
    # client = Client(local_cluster)
    #
    # print(f"local cluster: {local_cluster}")
    # print(f"client: {client}")

    output_filename = f"output_array.pkl"
    output_filepath = f"../resources/{output_filename}"

    output_chunk = (1, 61190)
    output_shape = (12000, 61190)

    print(f"starting...")

    start = time.time()

    if os.path.exists(output_filepath):
        sparse_da_training = load_da_array_pickle(output_filepath)

        print(f"{sparse_da_training}")
        print(f"{sparse_da_training.compute()}")

        print(f"chunk size: {sparse_da_training.chunksize}")

        PRINT_TEST_CALCULATIONS = True
    else:
        sparse_da_training = parse_data_training_array(
            f"../cs429529-project-2-topic-categorization/training.csv", output_filepath)

        PRINT_TEST_CALCULATIONS = False

    # 100 rows - 2-3 seconds
    # sparse_da_training = parse_data_training_bag_array_old(
    #     f"../cs429529-project-2-topic-categorization/training_small.csv")

    # 100 rows - 66-67 seconds
    # sparse_df_training = parse_data_training_bag_array_new(
    #     f"../cs429529-project-2-topic-categorization/training_small.csv")

    end = time.time()
    print(f"time elapsed: {end - start}")
    print(f"finished")

    ##################################################

    print(f"array size: {sparse_da_training.shape}")

    ##################################################

    if PRINT_TEST_CALCULATIONS:
        ##################################################

        print(f"test computation: sum along index")

        # timing comparison
        # (1, 61190) - 5-6 s
        # (5792, 5792) - 200-300 ms
        sparse_example = sparse_da_training.sum(axis=0)[0]

        print(f"{sparse_example.compute()}")

        ##################################################

        print(f"test computation: mat-vec mult.")

        # timing comparison
        # (1, 61190) - 5-6 s
        # (5792, 5792) - 200-300 ms
        sparse_example = da.matmul(sparse_da_training, da.ones((61190, )))
        sparse_example_output = sparse_example.compute()

        print(f"{sparse_example_output}")
        print(f"{type(sparse_example_output)}")

        ##################################################

        print(f"test computation: dot product")

        sparse_example = da.dot(sparse_da_training[:, 0], da.ones((12000, )))
        sparse_example_output = sparse_example.compute()

        # faster than using sum() above...
        print(f"{sparse_example_output}")
        print(f"{type(sparse_example_output)}")

        ##################################################

