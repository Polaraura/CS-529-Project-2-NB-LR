from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parameters.DataParameters import DataParameters
    from parameters.Hyperparameters import LogisticRegressionHyperparameters

from utilities.ParseUtilities import \
    get_data_from_file

import dask.bag as db
import dask.array as da
from dask.diagnostics import ProgressBar

from utilities.ParseUtilities import save_da_array_pickle, get_data_from_file
from utilities.DataFile import DataFileEnum
from utilities.ArrayUtilities import normalize_column_vector

from Constants import DELTA_MATRIX_FILEPATH, X_MATRIX_FILEPATH

import sparse
from math import exp
import numpy as np
from scipy.special import softmax


# TODO: remember to call .compute() AFTER all the iterations are completed


class LogisticRegression:
    def __init__(self,
                 input_array_da: da.array,
                 data_parameters: DataParameters,
                 hyperparameters: LogisticRegressionHyperparameters):
        # remove the index column
        self.input_array = input_array_da[:, 1:]
        self.data_parameters = data_parameters
        self.hyperparameters = hyperparameters

        # number of rows is the number of examples
        # number of columns is the total number of attributes (except for the index column at the beginning and the
        # class column at the end)
        self.m, self.n = self.input_array.shape

        # do not include the class column
        self.n -= 1

        self.k = len(data_parameters.class_labels_dict)

        self.class_vector = self.input_array[:, -1]
        self.delta_matrix = self.create_delta_matrix()

        # print(f"example delta matrix columns")
        # print(f"{self.delta_matrix[:, 0:20].compute()}")

        self.X_matrix = self.create_X_matrix()

        # check stats of X matrix...
        print(f"compute max X matrix...")
        print(f"{da.max(self.X_matrix).compute()}")

        self.Y_vector = self.create_Y_vector()
        self.W_matrix = self.create_W_matrix()

    def generate_delta_matrix(self):
        """
        Takes a while to construct the delta matrix because of the for loop...

        TODO: save to file
        :return:
        """

        class_column = self.class_vector
        num_data_rows = len(class_column)

        k, m = self.k, self.m
        delta_matrix = da.zeros((k, m), dtype=int)

        index_class_column = da.arange(num_data_rows).map_blocks(lambda x: sparse.COO(x, fill_value=0))

        print(f"class column: {class_column}")
        print(f"{class_column.compute().todense() - 1}")
        print(f"arange: {index_class_column}")

        # delta_matrix[enumerate(class_column)] = 1
        # class_column.compute().todense()
        # FIXME: need to subtract 1 for the class column (indexing starts at 0 instead of classification starting at 1)
        delta_matrix_rows = class_column.compute().todense() - 1
        delta_matrix_columns = da.arange(num_data_rows).compute()

        print(f"rows: {delta_matrix_rows}")
        print(f"rows: {delta_matrix_rows.shape}")
        print(f"columns: {delta_matrix_columns}")
        print(f"columns: {delta_matrix_columns.shape}")

        # FIXME: dask Array CANNOT do lists as indexing in MORE THAN 1 dimension
        # delta_matrix[delta_matrix_rows, delta_matrix_columns] = 1

        # TODO: implement as a standard for loop...
        for row_index, column_index in zip(delta_matrix_rows, delta_matrix_columns):
            delta_matrix[row_index, column_index] = 1

            # FIXME: max recursion depth error if all calculations are saved at the end...
            if (column_index + 1) % 100 == 0:
                print(f"saving delta matrix after {column_index + 1} iterations...")
                # delta_matrix = da.from_array(delta_matrix.compute())
                delta_matrix = delta_matrix.persist()

        # FIXME: rechunking doesn't change the chunksize...maybe too small
        # print(f"rechunk delta matrix...")
        # delta_matrix = delta_matrix.rechunk()

        # print(f"delta matrix before COO...")
        # delta_matrix.compute()

        # FIXME: don't convert to sparse...
        # delta_matrix = delta_matrix.map_blocks(lambda x: sparse.COO(x, fill_value=0))

        print(f"delta matrix: {delta_matrix}")

        save_da_array_pickle(delta_matrix, DELTA_MATRIX_FILEPATH)

        return delta_matrix

    def create_delta_matrix(self):
        return get_data_from_file(DataFileEnum.DELTA_MATRIX,
                                  self.generate_delta_matrix)

    def generate_X_matrix(self):
        """
        First column is all 1s (for initial weights w_0 when finding the probability matrix)
        Other columns are just values from the input data (excluding the class column)

        :return:
        """

        m, n = self.m, self.n
        X_matrix = da.zeros((m, n + 1), dtype=int)

        X_matrix[:, 0] = 1

        print(f"computing X matrix...")
        X_matrix[:, 1:] = self.input_array[:, :-1].compute().todense()

        # FIXME: need to map to COO BEFORE referencing input_array below since it is already COO...
        # FIXME: COO or sparse matrices in general doesn't support item assignment...
        X_matrix = X_matrix.map_blocks(lambda x: sparse.COO(x, fill_value=0))

        # print(f"input array: {self.input_array}")
        # print(f"input array: {self.input_array.compute()}")
        #
        # print(f"X matrix: intermediate steps...")
        # X_matrix = X_matrix.persist()

        save_da_array_pickle(X_matrix, X_MATRIX_FILEPATH)

        return X_matrix

    def create_X_matrix(self):
        return get_data_from_file(DataFileEnum.X_MATRIX,
                                  self.generate_X_matrix)

    def create_Y_vector(self):
        Y_vector = self.class_vector

        return Y_vector

    def create_W_matrix(self):
        """
        Initialize a weights matrix of zeros...

        TODO: initialize to very SMALL weights, NOT zeros...
        :return:
        """

        k, n = self.k, self.n
        W_matrix = da.zeros((k, n + 1), dtype=int)

        # FIXME: random initialization of W matrix...
        # mu = 0
        # sigma = 0.005
        #
        # # FIXME: fixed seed for testing...
        # np.random.seed(42)
        # da.random.seed(42)
        #
        # W_matrix = da.random.normal(mu, sigma, (k, n + 1))

        # need to map to sparse (to hold sparse results...)

        # FIXME: W won't be sparse after the 1st iteration...
        # W_matrix = W_matrix.map_blocks(lambda x: sparse.COO(x, fill_value=0))

        return W_matrix

    def normalize_W_matrix(self):
        """
        TODO: need to normalize the ENTIRE W matrix, not just along the column
        TODO: need to take abs first...

        :return:
        """

        # self.W_matrix = da.apply_along_axis(normalize_column_vector,
        #                                     1,
        #                                     self.W_matrix)\

        # print(f"total W matrix sum: {da.sum(da.abs(self.W_matrix)).compute()}")
        #
        # print(f"W matrix max BEFORE: {da.max(self.W_matrix).compute()}")

        self.W_matrix = self.W_matrix / da.sum(da.abs(self.W_matrix))
        self.W_matrix = self.W_matrix.persist()

        # print(f"W matrix max AFTER: {da.max(self.W_matrix).compute()}")

    def compute_probability_matrix(self):
        """
        P(Y | W, X) ~ exp(W X^T)

        Then fill in the last row with all 1s and normalize each column

        :return:
        """

        # print(f"computing max...")
        # print(f"max W matrix: {da.max(self.W_matrix).compute()}")

        # compute un-normalized probability matrix
        probability_Y_given_W_X_matrix_no_exp = da.dot(self.W_matrix,
                                                       da.transpose(self.X_matrix))

        # print(f"computing max...")
        # print(f"max no exp: {da.max(probability_Y_given_W_X_matrix_no_exp).compute()}")

        # FIXME
        # print(f"type of prob: {type(probability_Y_given_W_X_matrix_no_exp)}")
        # print(f"prob without exp: {probability_Y_given_W_X_matrix_no_exp[0][0:6].compute()}")

        # print(f"BEFORE EXP compute probability matrix...")
        # print(f"{self.W_matrix.compute()}")
        # print(f"{self.X_matrix.compute()}")
        # probability_Y_given_W_X_matrix_no_exp.compute()

        probability_Y_given_W_X_matrix = da.exp(probability_Y_given_W_X_matrix_no_exp)

        # FIXME
        # print(f"prob with exp: {probability_Y_given_W_X_matrix[0][0:6].compute()}")

        # print(f"probability matrix: {probability_Y_given_W_X_matrix}")
        # print(f"probability matrix: {probability_Y_given_W_X_matrix.compute()}")

        # print(f"compute probability matrix...")
        # probability_Y_given_W_X_matrix.compute()

        # set last row to all 1s
        probability_Y_given_W_X_matrix[-1, :] = 1

        # normalize each column
        normalized_probability_Y_given_W_X_matrix = da.apply_along_axis(normalize_column_vector,
                                                                        0,
                                                                        probability_Y_given_W_X_matrix)

        # FIXME: using softmax from scipy...similar results
        # print(f"normalizing probability matrix...")
        # normalized_probability_Y_given_W_X_matrix = softmax(probability_Y_given_W_X_matrix.compute(), axis=0)
        # normalized_probability_Y_given_W_X_matrix = da.from_array(normalized_probability_Y_given_W_X_matrix)

        # print(f"computing max...")
        # print(f"max prob: {da.max(normalized_probability_Y_given_W_X_matrix).compute()}")

        return normalized_probability_Y_given_W_X_matrix

    def compute_gradient_descent_step(self):
        hyperparameters = self.hyperparameters
        learning_rate = hyperparameters.learning_rate
        penalty_term = hyperparameters.penalty_term

        # print(f"W matrix: {self.W_matrix}")
        # print(f"delta matrix: {self.delta_matrix}")
        # print(f"X: matrix {self.X_matrix}")

        # FIXME: probability matrix doesn't have a fill value of 0...which conflicts with multiplication of other
        #  matrices with fill value of 0...
        # TODO: need to convert to dense matrix...
        # print(f"conversion of probability matrix to dense...")
        # probability_Y_given_W_X_matrix = da.exp(probability_Y_given_W_X_matrix_no_exp)
        # probability_Y_given_W_X_matrix = probability_Y_given_W_X_matrix.map_blocks(sparse.COO)

        probability_Y_given_W_X_matrix = self.compute_probability_matrix()

        # FIXME: for subtraction, also need to convert delta matrix to dense...
        intermediate_W_matrix = learning_rate * \
                                (da.dot((self.delta_matrix - probability_Y_given_W_X_matrix), self.X_matrix) -
                                 penalty_term * self.W_matrix)

        # FIXME
        # print(f"shape intermediate: {intermediate_W_matrix.shape}")
        # print(f"delta - P: {da.max(self.delta_matrix - probability_Y_given_W_X_matrix).compute()}")
        # print(f"(delta - P)X: "
        #       f"{da.max(da.dot((self.delta_matrix - probability_Y_given_W_X_matrix), self.X_matrix)).compute()}")
        # print(f"lambda W: {da.max(penalty_term * self.W_matrix).compute()}")
        # print(f"intermediate W: {da.max(intermediate_W_matrix).compute()}")

        # print(f"intermediate W: {intermediate_W_matrix.compute()}")
        # print(f"intermediate W dot: "
        #       f"{da.dot((self.delta_matrix - probability_Y_given_W_X_matrix), self.X_matrix).compute()}")

        self.W_matrix = self.W_matrix + \
                        intermediate_W_matrix

        # print(f"AFTER W matrix: {self.W_matrix}")

        # compute computation after EACH iteration
        # TODO: maybe experiment so computation is completed after a few iterations...

        print(f"calculating W matrix...")
        self.W_matrix = self.W_matrix.persist()
        # self.W_matrix = da.from_array(self.W_matrix.compute())

        print(f"normalizing W matrix...")
        self.normalize_W_matrix()

        # FIXME
        # print(f"computing example values of W matrix...")
        # print(f"final W shape: {self.W_matrix.shape}")
        # print(f"final W: {self.W_matrix[:][0:6].compute().T}")

    def complete_training(self):
        print(f"Starting training...")

        num_iter = self.hyperparameters.num_iter

        for i in range(num_iter):
            self.compute_gradient_descent_step()

            print(f"Iter {i} complete")

    def get_prediction(self,
                       data_row_da: da.array):
        # remember to remove the class column and substitute the index value with 1 in the data row
        print(f"convert data row to dense...")
        new_data_row = data_row_da[:-1].compute().todense()
        new_data_row[0] = 1

        prediction_vector = da.dot(self.W_matrix, new_data_row)

        data_row_argmax = da.argmax(prediction_vector)

        print(f"prediction vector: {prediction_vector.compute()}")

        print(f"find max of prediction...")
        data_row_argmax = data_row_argmax.compute()

        print(f"argmax: {data_row_argmax}")

        return data_row_argmax + 1

    def set_initial_parameters(self):
        pass


if __name__ == "__main__":
    # prepare progress bar
    pbar = ProgressBar()
    pbar.register()  # global registration

    a = np.array([[1, 2, 3], [4, 5, 6]])

    print(f"{a[np.array([0, 1]), np.array([0, 1])]}")

    c = da.zeros((1000, 1000)).map_blocks(sparse.COO)
    d = da.zeros((5000, 1000))

    # c[0, 0] = 1
    # d[0, 0] = 1

    b = da.dot(c, da.transpose(d))

    b = b - 5

    print(f"{b.compute()}")

    e = da.array(np.array([0, 0, 1, 2])).map_blocks(sparse.COO)
    print(f"{da.exp(e).compute()}")
    print(f"{da.exp(e).compute().todense()}")
