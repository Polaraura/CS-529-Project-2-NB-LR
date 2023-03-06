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

import zarr
import h5py
import pickle

import sparse

from Constants import CHUNK_SIZE


def parse_data_training_dataframe(filename, chunksize=CHUNK_SIZE):
    # data = ddf.read_csv(filename, header=None)
    print(f"initial read text")
    data = ddf.read_csv(filename, blocksize='10MB')

    print(f"map to numpy array")
    data = data.map(lambda x: np.loadtxt(StringIO(x), delimiter=','))
    data = data.map(lambda x: SparseArray(x, fill_value=0))


def parse_data_training_bag(filename, chunksize=CHUNK_SIZE):
    # data = ddf.read_csv(filename, header=None)
    print(f"initial read text")
    data = db.read_text(filename, blocksize='10MB')

    print(f"map to numpy array")
    data = data.map(lambda x: np.loadtxt(StringIO(x), delimiter=','))
    # print(f"{data}")

    # data_print = data.compute()
    # print(f"{data_print}")

    # data_array = data.to_dask_array(lengths=True)
    # data_array = da.concatenate(data, axis=0)
    print(f"map to dask array")
    data = data.map(lambda x: da.from_array(x))
    print(f"map to sparse dask array")
    data_array = data.map(lambda x: x.map_blocks(lambda x: sparse.COO(x, fill_value=0)))

    print(f"{data_array}")

    sparse_chunks = data_array

    print(f"reduce sparse dask array")

    return sparse_chunks


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

    # sparse_example = sparse_df_training.rechunk((1000, 1000))
    # print(f"test computation")
    # sparse_example = sparse_chunks.sum(axis=0)[0]
    #
    # print(f"{sparse_example.compute()}")

    with open(output_filepath, 'wb') as output_file:
        # FIXME: can't pickle the lazy object...
        # pickle.dump(sparse_chunks, output_file)
        pickle.dump(sparse_chunks.compute(), output_file)

    return sparse_chunks

    ################################################

    # sparse_example = sparse_chunks.map(lambda x: x.sum(axis=0).compute())
    # a = sparse_example.compute()
    #
    # print(f"{a}")
    #
    # return sparse_chunks


def parse_data_training_bag_array_new(filename, chunksize=CHUNK_SIZE):
    # data = ddf.read_csv(filename, header=None)

    # only reads line by line...
    print(f"initial read text")
    data = ddf.read_csv(filename, blocksize="100MB")

    print(f"to dask array")
    data = data.to_dask_array()

    print(f"map to sparse array")
    data = data.map_blocks(lambda x: sparse.COO(x, fill_value=0))

    print(f"test computation")

    # a = data.sum(axis=1).compute()
    # print(f"{a.todense()}")

    a = data.sum(axis=0)[0]
    print(f"{a.compute()}")




def parse_data_training_pandas(filename, chunksize=CHUNK_SIZE):
    # data = ddf.read_csv(filename, header=None)
    data = pd.read_csv(filename, header=None)
    data = ddf.from_pandas(data, chunksize=100)
    # print(f"{data}")

    # data_array = data.to_dask_array(lengths=True)
    data_array = data.to_dask_array()

    print(f"{data_array}")

    sparse_chunks = data_array.map_blocks(lambda x: sparse.COO(x, fill_value=0))
    sparse_chunks = sparse_chunks.compute_chunk_sizes()

    print(f"{sparse_chunks}")

    # sparse_example = sparse_df_training.rechunk((1000, 1000))
    sparse_example = sparse_chunks.sum(axis=1)[5]

    print(f"{sparse_example.compute()}")

    return sparse_chunks


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
    sparse_example = sparse_da_training.sum(axis=0)[0]

    print(f"{sparse_example.compute()}")

    ##################################################

    # print(pd.arrays.SparseArray(list(np.array([[1, 2], [3, 4]]))))

    # chunk_size_list = [100, 200, 500, 1000, 2000, 5000]
    #
    # for chunk_size in chunk_size_list:
    #     start = time.time()
    #     parse_data_training(f"../cs429529-project-2-topic-categorization/training.csv", chunksize=chunk_size)
    #     end = time.time()
    #
    #     print(f"chunk size: {chunk_size}, time elapsed: {end - start}")

