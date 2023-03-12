from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parameters.DataParameters import DataParameters
    from parameters.Hyperparameters import LogisticRegressionHyperparameters

import dask.bag as db
import dask.array as da

import sparse
from math import exp

# TODO: remember to call .compute() AFTER all the iterations are completed


class LogisticRegression:
    def __init__(self,
                 input_array: da.array,
                 data_parameters: DataParameters,
                 hyperparameters: LogisticRegressionHyperparameters):
        # remove the index column
        self.input_array = input_array[:, 1:]
        self.data_parameters = data_parameters
        self.hyperparameters = hyperparameters

        # number of rows is the number of examples
        # number of columns is the total number of attributes (except for the index column at the beginning and the
        # class column at the end)
        self.m, self.n = self.input_array

        # do not include the class column
        self.n -= 1

        self.k = len(data_parameters.class_labels_dict)

        self.class_vector = self.input_array[:, -1]
        self.delta_matrix = self.create_delta_matrix()
        self.X_matrix = self.create_X_matrix()
        self.Y_vector = self.create_Y_vector()
        self.W_matrix = self.create_W_matrix()

    def create_delta_matrix(self):
        class_column = self.class_vector

        k, m = self.k, self.m
        delta_matrix = da.zeros((k, m))

        delta_matrix[enumerate(class_column)] = 1

        delta_matrix.map_blocks(lambda x: sparse.COO(x, fill_value=0))

        return delta_matrix

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

        probability_Y_given_W_X_matrix = exp(da.matmul(self.W_matrix,
                                                       da.transpose(self.X_matrix)))

        self.W_matrix = self.W_matrix + \
                        learning_rate * \
                        (da.matmul((self.delta_matrix - probability_Y_given_W_X_matrix), self.X_matrix) -
                         penalty_term * da.transpose(self.W_matrix))


    def set_initial_parameters(self):
        pass
