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

from Constants import CHUNK_SIZE


def parse_data_training_bag_array_old(filename, output_filepath, chunksize=CHUNK_SIZE):
    # data = ddf.read_csv(filename, header=None)

    # only reads line by line...
    print(f"initial read text")
    data = db.read_text(filename, blocksize="10MB", linedelimiter="\n")
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

    with open(output_filepath, 'wb') as output_file:
        # FIXME: can't pickle the lazy object...
        # pickle.dump(sparse_chunks, output_file)
        pickle.dump(sparse_chunks.compute(), output_file)

    return sparse_chunks

    ################################################


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

    start = time.time()

    if os.path.exists(output_filepath):
        output_file = open(output_filepath, "rb")

        # computed object...
        # IMPORTANT: chunk size is set to auto (default optional arg)
        # initially read in row by row with chunk size (1, 61190) -- pretty slow
        # auto set chunk size to (5792, 5792) -- much faster (AFTER .pkl file was created)
        # see test computation below for timings comparison
        sparse_da_training = pickle.load(output_file)
        sparse_da_training = da.from_array(sparse_da_training)

        print(f"{sparse_da_training}")
        print(f"{sparse_da_training.compute()}")
    else:
        sparse_da_training = parse_data_training_bag_array_old(
            f"../cs429529-project-2-topic-categorization/training.csv", output_filepath)

    print(f"starting...")

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

    print(f"test computation")

    # timing comparison
    # (1, 61190) - 5-6 s
    # (5792, 5792) - 200-300 ms
    sparse_example = sparse_da_training.sum(axis=0)[0]

    print(f"{sparse_example.compute()}")

    ##################################################

