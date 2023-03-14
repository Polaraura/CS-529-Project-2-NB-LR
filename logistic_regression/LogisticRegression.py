from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from parameters.DataParameters import DataParameters
    from parameters.Hyperparameters import LogisticRegressionHyperparameters

from utilities.ParseUtilities import \
    get_data_from_file

import dask.bag as db
import dask.array as da

from utilities.ParseUtilities import save_da_array_pickle, get_data_from_file
from utilities.DataFile import DataFileEnum

from Constants import DELTA_MATRIX_FILEPATH

import sparse
from math import exp
import numpy as np

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
        self.X_matrix = self.create_X_matrix()
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
        delta_matrix = da.zeros((k, m))

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
                delta_matrix = da.from_array(delta_matrix.compute())

        print(f"delta matrix before COO...")
        delta_matrix.compute()

        delta_matrix.map_blocks(lambda x: sparse.COO(x, fill_value=0))

        print(f"delta matrix: {delta_matrix}")

        save_da_array_pickle(delta_matrix, DELTA_MATRIX_FILEPATH)

        return delta_matrix

    def create_delta_matrix(self):
        return get_data_from_file(DataFileEnum.DELTA_MATRIX,
                                  self.generate_delta_matrix)

    def create_X_matrix(self):
        m, n = self.m, self.n
        X_matrix = da.zeros((m, n + 1))

        X_matrix[:, 0] = 1
        X_matrix[:, 1:] = self.input_array[:, :-1]
        X_matrix.map_blocks(lambda x: sparse.COO(x, fill_value=0))

        return X_matrix

    def create_Y_vector(self):
        Y_vector = self.class_vector

        return Y_vector

    def create_W_matrix(self):
        """
        Initialize a weights matrix of zeros...

        :return:
        """

        k, n = self.k, self.n
        W_matrix = da.zeros((k, n + 1))

        return W_matrix

    def compute_gradient_descent_step(self):
        hyperparameters = self.hyperparameters
        learning_rate = hyperparameters.learning_rate
        penalty_term = hyperparameters.penalty_term

        probability_Y_given_W_X_matrix_no_exp = da.matmul(self.W_matrix,
                                                       da.transpose(self.X_matrix))
        probability_Y_given_W_X_matrix = exp(probability_Y_given_W_X_matrix_no_exp)

        self.W_matrix = self.W_matrix + \
                        learning_rate * \
                        (da.matmul((self.delta_matrix - probability_Y_given_W_X_matrix), self.X_matrix) -
                         penalty_term * da.transpose(self.W_matrix))

    def complete_training(self):
        print(f"Starting training...")

        num_iter = self.hyperparameters.num_iter

        for i in range(num_iter):
            self.compute_gradient_descent_step()

            print(f"Iter {i} complete")

    def get_prediction(self,
                       data_row_da: da.array):
        data_row_argmax = da.argmax(da.matmul(self.W_matrix, data_row_da))
        data_row_argmax = data_row_argmax.compute()

        print(f"argmax: {data_row_argmax}")

        return data_row_argmax + 1

    def set_initial_parameters(self):
        pass


if __name__ == "__main__":
    a = numpy.array([[1, 2, 3], [4, 5, 6]])

    print(f"{a[np.array([0, 1]), np.array([0, 1])]}")

