from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parameters.DataParameters import DataParameters
    from parameters.Hyperparameters import LogisticRegressionHyperparameters

from utilities.ParseUtilities import \
    get_data_from_file, create_sub_directories

import dask.bag as db
import dask.array as da
from dask.diagnostics import ProgressBar

from utilities.ParseUtilities import save_da_array_pickle, get_data_from_file
from utilities.DataFile import DataFileEnum, DataOptionEnum, WMatrixOptionEnum
from utilities.ArrayUtilities import normalize_column_vector

from Constants import DELTA_MATRIX_FILEPATH, W_MATRIX_FILENAME_WITHOUT_EXTENSION, \
    W_MATRIX_FILENAME_EXTENSION, OUTPUT_DIR, DIR_UP, W_MATRIX_TOP_DIR, W_MATRIX_SUB_DIR_DICT, PATH_SEP, \
    X_MATRIX_SUB_DIR_DICT, X_MATRIX_FILEPATH_NO_NORMALIZATION, X_MATRIX_FILEPATH_X_NORMALIZATION

from utilities.DebugFlags import LOGISTIC_REGRESSION_DELTA_DEBUG, LOGISTIC_REGRESSION_X_MATRIX_DEBUG, \
    LOGISTIC_REGRESSION_NORMALIZE_W_MATRIX_DEBUG, LOGISTIC_REGRESSION_PROBABILITY_MATRIX_DEBUG, \
    LOGISTIC_REGRESSION_GRADIENT_DESCENT_DEBUG, LOGISTIC_REGRESSION_TRAINING_DEBUG, \
    LOGISTIC_REGRESSION_PREDICTION_DEBUG

import sparse
from math import exp
import numpy as np
from scipy.special import softmax

import time
import glob
import re
import os
from pathlib import Path


class LogisticRegression:
    def __init__(self,
                 input_array_da: da.array,
                 data_parameters: DataParameters,
                 hyperparameters: LogisticRegressionHyperparameters,
                 normalize_W=True,
                 normalize_X=False):
        # remove the index column
        self.input_array = input_array_da[:, 1:]
        self.data_parameters = data_parameters
        self.hyperparameters = hyperparameters
        self.normalize_W = normalize_W
        self.normalize_X = normalize_X

        # set execution filepath of X matrix
        self.X_matrix_filepath = None

        # set execution filepath of W matrix
        self.output_dir = f"{OUTPUT_DIR}"
        self.filepath_W_matrix = self.get_filepath_W_matrix(DataOptionEnum.LOAD)
        self.filepath_W_matrix_normalization_string = None
        self.W_matrix_parent_dir = None

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

        self.Y_vector = self.create_Y_vector()
        self.W_matrix = self.create_W_matrix()

    def generate_delta_matrix(self):
        """
        Takes a while to construct the delta matrix because of the for loop...

        :return:
        """

        class_column = self.class_vector
        num_data_rows = len(class_column)

        k, m = self.k, self.m
        delta_matrix = da.zeros((k, m), dtype=int)

        index_class_column = da.arange(num_data_rows).map_blocks(lambda x: sparse.COO(x, fill_value=0))

        if LOGISTIC_REGRESSION_DELTA_DEBUG:
            print(f"class column: {class_column}")
            print(f"{class_column.compute().todense() - 1}")
            print(f"arange: {index_class_column}")

        # need to subtract 1 for the class column (indexing starts at 0 instead of classification starting at 1)
        delta_matrix_rows = class_column.compute().todense() - 1
        delta_matrix_columns = da.arange(num_data_rows).compute()

        if LOGISTIC_REGRESSION_DELTA_DEBUG:
            print(f"rows: {delta_matrix_rows}")
            print(f"rows: {delta_matrix_rows.shape}")
            print(f"columns: {delta_matrix_columns}")
            print(f"columns: {delta_matrix_columns.shape}")

        # FIXME: dask Array CANNOT do lists as indexing in MORE THAN 1 dimension
        # delta_matrix[delta_matrix_rows, delta_matrix_columns] = 1

        for row_index, column_index in zip(delta_matrix_rows, delta_matrix_columns):
            delta_matrix[row_index, column_index] = 1

            # max recursion depth error if all calculations are saved at the end...
            if (column_index + 1) % 100 == 0:
                print(f"saving delta matrix after {column_index + 1} iterations...")
                # delta_matrix = da.from_array(delta_matrix.compute())
                delta_matrix = delta_matrix.persist()

        # rechunking doesn't change the chunksize...maybe too small
        # print(f"rechunk delta matrix...")
        # delta_matrix = delta_matrix.rechunk()

        save_da_array_pickle(delta_matrix, DELTA_MATRIX_FILEPATH)

        return delta_matrix

    def create_delta_matrix(self):
        return get_data_from_file(DataFileEnum.DELTA_MATRIX, self.generate_delta_matrix)

    def generate_X_matrix(self):
        """
        First column is all 1s (for initial weights w_0 when finding the probability matrix)
        Other columns are just values from the input data (excluding the class column)

        :return:
        """

        m, n = self.m, self.n
        X_matrix = da.zeros((m, n + 1), dtype=int)

        X_matrix[:, 0] = 1

        if LOGISTIC_REGRESSION_X_MATRIX_DEBUG:
            print(f"computing X matrix...")

        # FIXME: expensive operation...
        X_matrix[:, 1:] = self.input_array[:, :-1].compute().todense()

        # TODO: check for normalization
        if self.normalize_X:
            # normalize each column (along the feature/word instead of each instance)
            X_matrix = da.apply_along_axis(normalize_column_vector,
                                           0,
                                           X_matrix)

        # COO or sparse matrices in general doesn't support item assignment...
        X_matrix = X_matrix.map_blocks(lambda x: sparse.COO(x, fill_value=0))

        if LOGISTIC_REGRESSION_X_MATRIX_DEBUG:
            print(f"input array: {self.input_array}")
            print(f"input array: {self.input_array.compute()}")

            # print(f"X matrix: intermediate steps...")
            # X_matrix = X_matrix.persist()

        # need to create sub_directories first
        # FIXME: don't include the filename...
        X_matrix_filepath_path = Path(self.X_matrix_filepath)
        X_matrix_parent_path = X_matrix_filepath_path.parent

        create_sub_directories(str(X_matrix_parent_path))

        save_da_array_pickle(X_matrix, self.X_matrix_filepath)

        return X_matrix

    def create_X_matrix(self):
        # TODO: edit file path (depending on normalization)
        if self.normalize_X:
            X_matrix_filepath = X_MATRIX_FILEPATH_X_NORMALIZATION
        else:
            X_matrix_filepath = X_MATRIX_FILEPATH_NO_NORMALIZATION

        self.X_matrix_filepath = X_matrix_filepath

        X_matrix = get_data_from_file(DataFileEnum.X_MATRIX,
                                      self.generate_X_matrix,
                                      custom_filepath=X_matrix_filepath)

        if LOGISTIC_REGRESSION_X_MATRIX_DEBUG:
            # check stats of X matrix...
            print(f"compute max X matrix...")
            print(f"{da.max(X_matrix).compute()}")

        return X_matrix

    def create_Y_vector(self):
        Y_vector = self.class_vector

        return Y_vector

    def get_W_matrix_sub_dir(self):
        sub_dir = f""

        if self.normalize_W or self.normalize_X:
            if self.normalize_W and self.normalize_X:
                sub_dir += f"{W_MATRIX_SUB_DIR_DICT[WMatrixOptionEnum.W_X_NORMALIZATION]}"
            elif self.normalize_W:
                sub_dir += f"{W_MATRIX_SUB_DIR_DICT[WMatrixOptionEnum.W_NORMALIZATION]}"
            elif self.normalize_X:
                sub_dir += f"{W_MATRIX_SUB_DIR_DICT[WMatrixOptionEnum.X_NORMALIZATION]}"
        else:
            sub_dir += f"{W_MATRIX_SUB_DIR_DICT[WMatrixOptionEnum.NO_NORMALIZATION]}"

        return sub_dir

    def get_filepath_iter_num_key_W_matrix(self, filepath: str):
        """
        TODO: split and re-add the sep regex
        :param filepath:
        :return:
        """

        extra_string = self.filepath_W_matrix_normalization_string
        W_matrix_sub_dir = self.get_W_matrix_sub_dir()

        dir_sep_regex = fr"\{PATH_SEP}"

        # FIXME: filepaths in regex...need to escape dir separators
        # output_dir, W_MATRIX_TOP_DIR, W_matrix_sub_dir
        # iter_num_regex = re.compile(fr"{self.output_dir}{dir_sep_regex}"
        #                             fr"{W_MATRIX_TOP_DIR}{dir_sep_regex}"
        #                             fr"{W_matrix_sub_dir}{dir_sep_regex}"
        #                             fr"{W_MATRIX_FILENAME_WITHOUT_EXTENSION}"
        #                             fr"{extra_string}-(\d+)\.pkl")
        W_matrix_parent_dir_split = self.W_matrix_parent_dir.split(PATH_SEP)
        W_matrix_parent_dir_regex = dir_sep_regex.join(W_matrix_parent_dir_split)
        iter_num_regex = re.compile(fr"{W_matrix_parent_dir_regex}{dir_sep_regex}"
                                    fr"{W_MATRIX_FILENAME_WITHOUT_EXTENSION}"
                                    fr"{extra_string}-(\d+)\.pkl")

        iter_num_match = re.search(iter_num_regex, filepath)

        if iter_num_match is None:
            raise ValueError("no path...")

        # return the first group with the iter_num (cast to int)
        return int(iter_num_match.group(1))

    def get_filepath_W_matrix(self, data_option: DataOptionEnum):
        """
        format of filepath

        w_matrix - top level dir
        |- no_normalization - sub dir
        |- w_normalization - sub dir
        |- x_normalization - sub dir
        |- w_x_normalization - sub dir

        file name
        w_matrix-<iter_num>.pkl

        e.g., w_matrix-10.pkl (after a total of 10 iterations, the corresponding W matrix was saved)

        will continue on from the most recent (i.e., file with the highest iter_num) on next execution

        iter_num - the collective amount of iterations (gradient descent) trained on the W matrix

        :return:
        """

        # if BOTH normalize options are set to False
        iter_string = f""

        # if self.normalize_W or self.normalize_X:
        #     iter_string = f"_normalize_"
        #     if self.normalize_W and self.normalize_X:
        #         iter_string += f"wx"
        #     elif self.normalize_W:
        #         iter_string += f"w"
        #     elif self.normalize_X:
        #         iter_string += f"x"

        output_dir = self.output_dir
        W_matrix_sub_dir = self.get_W_matrix_sub_dir()
        W_matrix_parent_dir = os.path.join(output_dir, W_MATRIX_TOP_DIR, W_matrix_sub_dir)

        self.W_matrix_parent_dir = W_matrix_parent_dir

        # TODO: need to find most recent iteration num
        # fr"{W_matrix_parent_dir}{W_MATRIX_FILENAME_WITHOUT_EXTENSION}{iter_string}-*"
        W_matrix_filenames_regex = os.path.join(W_matrix_parent_dir,
                                                fr"{W_MATRIX_FILENAME_WITHOUT_EXTENSION}-*")
        W_matrix_filenames = glob.glob(W_matrix_filenames_regex)

        self.filepath_W_matrix_normalization_string = iter_string

        # not empty
        if W_matrix_filenames:
            # assume list is NOT sorted - ONLY when the list is NOT empty
            sorted_W_matrix_filenames = sorted(W_matrix_filenames, key=self.get_filepath_iter_num_key_W_matrix,
                                               reverse=True)

            print(f"sorted W matrix filenames: {sorted_W_matrix_filenames}")

            prev_iter_num = self.get_filepath_iter_num_key_W_matrix(sorted_W_matrix_filenames[0])
        # empty
        else:
            prev_iter_num = 0

        if data_option == DataOptionEnum.SAVE:
            next_iter_num = self.hyperparameters.num_iter
            next_iter_num += prev_iter_num
            iter_string += f"-{next_iter_num}"
        elif data_option == DataOptionEnum.LOAD:
            iter_string += f"-{prev_iter_num}"
        else:
            raise ValueError("Invalid option to get W matrix filepath...\nSave or Load\n")

        # # recursively create sub directories
        # W_matrix_parent_dir_path = Path(W_matrix_parent_dir)
        # W_matrix_parent_dir_path.mkdir(parents=True, exist_ok=True)

        create_sub_directories(W_matrix_parent_dir)

        # get filename
        W_matrix_filename = f"{W_MATRIX_FILENAME_WITHOUT_EXTENSION}{iter_string}{W_MATRIX_FILENAME_EXTENSION}"
        # w_matrix_filepath = f"{output_dir}{W_MATRIX_TOP_DIR}{W_matrix_sub_dir}" \
        #                     f"{W_MATRIX_FILENAME_WITHOUT_EXTENSION}{iter_string}" \
        #                     f"{W_MATRIX_FILENAME_EXTENSION}"

        # create path for W matrix file
        W_matrix_filepath = os.path.join(W_matrix_parent_dir, W_matrix_filename)

        return W_matrix_filepath

    def generate_W_matrix(self,
                          random_initial=False):
        """
        Initialize a weights matrix of zeros...

        :param random_initial:
        :return:
        """

        k, n = self.k, self.n

        if not random_initial:
            W_matrix = da.zeros((k, n + 1), dtype=int)
        else:
            # random initialization of W matrix
            mu = 0
            sigma = 0.005

            # fixed seed for testing
            np.random.seed(42)
            da.random.seed(42)

            W_matrix = da.random.normal(mu, sigma, (k, n + 1))

        # W won't be sparse after the 1st iteration...multiplication with X matrix
        # W_matrix = W_matrix.map_blocks(lambda x: sparse.COO(x, fill_value=0))

        return W_matrix

    def create_W_matrix(self):
        return get_data_from_file(DataFileEnum.W_MATRIX,
                                  self.generate_W_matrix,
                                  custom_filepath=self.filepath_W_matrix)

    def normalize_W_matrix(self):
        """
        need to normalize the ENTIRE W matrix, not just along the column
        need to take abs first...

        :return:
        """

        # do not normalize along a column/row
        # self.W_matrix = da.apply_along_axis(normalize_column_vector,
        #                                     1,
        #                                     self.W_matrix)\

        if LOGISTIC_REGRESSION_NORMALIZE_W_MATRIX_DEBUG:
            print(f"total W matrix sum: {da.sum(da.abs(self.W_matrix)).compute()}")
            print(f"W matrix max BEFORE: {da.max(self.W_matrix).compute()}")

        self.W_matrix = self.W_matrix / da.sum(da.abs(self.W_matrix))
        self.W_matrix = self.W_matrix.persist()

        if LOGISTIC_REGRESSION_NORMALIZE_W_MATRIX_DEBUG:
            print(f"W matrix max AFTER: {da.max(self.W_matrix).compute()}")

    def get_W_matrix(self):
        return self.W_matrix.compute()

    def set_W_matrix(self, W_matrix):
        self.W_matrix = W_matrix

    def compute_probability_matrix(self):
        """
        P(Y | W, X) ~ exp(W X^T)

        Then fill in the last row with all 1s and normalize each column

        probability matrix cannot be a sparse matrix
        probability matrix may not have a fill value of 0...which conflicts with multiplication of other matrices
        with fill value of 0 (subtraction with delta matrix in gradient descent step)

        :return:
        """

        if LOGISTIC_REGRESSION_PROBABILITY_MATRIX_DEBUG:
            print(f"computing max...")
            print(f"max W matrix: {da.max(self.W_matrix).compute()}")

        # compute un-normalized probability matrix
        probability_Y_given_W_X_matrix_no_exp = da.dot(self.W_matrix,
                                                       da.transpose(self.X_matrix))

        if LOGISTIC_REGRESSION_PROBABILITY_MATRIX_DEBUG:
            print(f"computing max...")
            print(f"max no exp: {da.max(probability_Y_given_W_X_matrix_no_exp).compute()}")

            print(f"type of prob: {type(probability_Y_given_W_X_matrix_no_exp)}")
            print(f"prob without exp: {probability_Y_given_W_X_matrix_no_exp[0][0:6].compute()}")

            print(f"BEFORE EXP compute probability matrix...")
            print(f"{self.W_matrix.compute()}")
            print(f"{self.X_matrix.compute()}")
            probability_Y_given_W_X_matrix_no_exp.compute()

        probability_Y_given_W_X_matrix = da.exp(probability_Y_given_W_X_matrix_no_exp)

        if LOGISTIC_REGRESSION_PROBABILITY_MATRIX_DEBUG:
            print(f"prob with exp: {probability_Y_given_W_X_matrix[0][0:6].compute()}")

            print(f"probability matrix: {probability_Y_given_W_X_matrix}")
            print(f"probability matrix: {probability_Y_given_W_X_matrix.compute()}")

            print(f"compute probability matrix...")
            probability_Y_given_W_X_matrix.compute()

        # set last row to all 1s
        probability_Y_given_W_X_matrix[-1, :] = 1

        # normalize each column
        normalized_probability_Y_given_W_X_matrix = da.apply_along_axis(normalize_column_vector,
                                                                        0,
                                                                        probability_Y_given_W_X_matrix)

        # using softmax from scipy...similar results
        # print(f"normalizing probability matrix...")
        # normalized_probability_Y_given_W_X_matrix = softmax(probability_Y_given_W_X_matrix.compute(), axis=0)
        # normalized_probability_Y_given_W_X_matrix = da.from_array(normalized_probability_Y_given_W_X_matrix)

        # print(f"computing max...")
        # print(f"max prob: {da.max(normalized_probability_Y_given_W_X_matrix).compute()}")

        return normalized_probability_Y_given_W_X_matrix

    def compute_probability_vector_prediction(self,
                                              data_row_da: da.array):
        """
        P(Y | W, X) ~ exp(W X^T)

        Similar to above, but on a particular data row

        :return:
        """

        probability_Y_given_W_X_vector_no_exp = da.dot(self.W_matrix,
                                                       data_row_da)

        probability_Y_given_W_X_vector = da.exp(probability_Y_given_W_X_vector_no_exp)

        # set last row to all 1s
        probability_Y_given_W_X_vector[-1] = 1

        # normalize each column
        normalized_probability_Y_given_W_X_vector = normalize_column_vector(probability_Y_given_W_X_vector)

        return normalized_probability_Y_given_W_X_vector

    def compute_gradient_descent_step(self):
        hyperparameters = self.hyperparameters
        learning_rate = hyperparameters.learning_rate
        penalty_term = hyperparameters.penalty_term

        if LOGISTIC_REGRESSION_GRADIENT_DESCENT_DEBUG:
            print(f"W matrix: {self.W_matrix}")
            print(f"delta matrix: {self.delta_matrix}")
            print(f"X: matrix {self.X_matrix}")

        probability_Y_given_W_X_matrix = self.compute_probability_matrix()

        intermediate_W_matrix = learning_rate * \
                                (da.dot((self.delta_matrix - probability_Y_given_W_X_matrix), self.X_matrix) -
                                 penalty_term * self.W_matrix)

        if LOGISTIC_REGRESSION_GRADIENT_DESCENT_DEBUG:
            print(f"shape intermediate: {intermediate_W_matrix.shape}")
            print(f"delta - P: {da.max(self.delta_matrix - probability_Y_given_W_X_matrix).compute()}")
            print(f"(delta - P)X: "
                  f"{da.max(da.dot((self.delta_matrix - probability_Y_given_W_X_matrix), self.X_matrix)).compute()}")
            print(f"lambda W: {da.max(penalty_term * self.W_matrix).compute()}")
            print(f"intermediate W: {da.max(intermediate_W_matrix).compute()}")

            print(f"intermediate W: {intermediate_W_matrix.compute()}")
            print(f"intermediate W dot: "
                  f"{da.dot((self.delta_matrix - probability_Y_given_W_X_matrix), self.X_matrix).compute()}")

        self.W_matrix = self.W_matrix + \
                        intermediate_W_matrix

        if LOGISTIC_REGRESSION_GRADIENT_DESCENT_DEBUG:
            print(f"AFTER W matrix: {self.W_matrix}")

        # compute computation after EACH iteration
        # print(f"calculating W matrix...")
        self.W_matrix = self.W_matrix.persist()
        # self.W_matrix = da.from_array(self.W_matrix.compute())

        if self.normalize_W:
            # print(f"normalizing W matrix...")
            self.normalize_W_matrix()

        if LOGISTIC_REGRESSION_GRADIENT_DESCENT_DEBUG:
            print(f"computing example values of W matrix...")
            print(f"final W shape: {self.W_matrix.shape}")
            print(f"final W: {self.W_matrix[:][0:6].compute().T}")

    def complete_training(self):
        print(f"Starting training...")

        num_iter = self.hyperparameters.num_iter

        if num_iter == 0:
            print(f"No training done")
            return

        for i in range(num_iter):
            self.compute_gradient_descent_step()

            if LOGISTIC_REGRESSION_TRAINING_DEBUG:
                print(f"Iter {i} complete")

        if LOGISTIC_REGRESSION_TRAINING_DEBUG:
            # finish W matrix
            print(f"saving W matrix...")
            start_time = time.time()

        # self.W_matrix = self.W_matrix.compute()

        # TODO: save W matrix into file
        # TODO: find previous iteration num among files...should work if files are deleted
        save_da_array_pickle(self.W_matrix, self.get_filepath_W_matrix(DataOptionEnum.SAVE))

        if LOGISTIC_REGRESSION_TRAINING_DEBUG:
            end_time = time.time()
            print(f"W matrix time: {end_time - start_time}")

    def get_prediction(self,
                       data_row_da: da.array):
        if LOGISTIC_REGRESSION_PREDICTION_DEBUG:
            print(f"convert data row to dense...")

        # remember to remove the class column and substitute the index value with 1 in the data row
        processed_data_row = data_row_da[:-1].compute().todense()
        processed_data_row[0] = 1

        if LOGISTIC_REGRESSION_PREDICTION_DEBUG:
            print(f"W matrix shape: {self.W_matrix.shape}")
            print(f"data row shape: {processed_data_row.shape}")

        # probability_prediction_vector = da.dot(self.W_matrix, new_data_row)
        probability_prediction_vector = self.compute_probability_vector_prediction(processed_data_row)

        if LOGISTIC_REGRESSION_PREDICTION_DEBUG:
            print(f"computing argmax...")

        data_row_argmax = da.argmax(probability_prediction_vector)
        data_row_argmax = data_row_argmax.compute()

        if LOGISTIC_REGRESSION_PREDICTION_DEBUG:
            print(f"computing entire prediction vector...")
            print(f"prediction vector: {probability_prediction_vector.compute()}")
            print(f"find max of prediction...")
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
