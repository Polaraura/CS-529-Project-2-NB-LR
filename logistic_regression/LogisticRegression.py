from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parameters.DataParameters import DataParameters
    from parameters.Hyperparameters import LogisticRegressionHyperparameters

from utilities.ParseUtilities import \
    get_data_from_file, parse_data_training_testing_array, load_da_array_pickle
from utilities.FileSystemUtilities import create_sub_directories

import dask.bag as db
import dask.array as da
import dask.dataframe as df
from dask.diagnostics import ProgressBar

import pandas as pd

from utilities.ParseUtilities import save_da_array_pickle, get_data_from_file
from utilities.DataFile import DataFileEnum, DataOptionEnum, WMatrixOptionEnum, XMatrixType, ClassVectorType
from utilities.ArrayUtilities import normalize_column_vector, calculate_entropy_column_vector, \
    calculate_gini_index_column_vector

from Constants import DELTA_MATRIX_FILEPATH, W_MATRIX_FILENAME_WITHOUT_EXTENSION, \
    W_MATRIX_FILENAME_EXTENSION, OUTPUT_DIR, DIR_UP, W_MATRIX_TOP_DIR, W_MATRIX_SUB_DIR_DICT, PATH_SEP, \
    X_MATRIX_SUB_DIR_DICT, X_MATRIX_FILEPATH_NO_NORMALIZATION_TRAINING, X_MATRIX_FILEPATH_X_NORMALIZATION_TRAINING, \
    X_MATRIX_FILEPATH_X_NORMALIZATION_VALIDATION, X_MATRIX_FILEPATH_NO_NORMALIZATION_VALIDATION, \
    INPUT_ARRAY_FILEPATH_TRAINING, INPUT_ARRAY_FILEPATH_VALIDATION, X_MATRIX_FILEPATH_NO_NORMALIZATION_TESTING, \
    X_MATRIX_FILEPATH_X_NORMALIZATION_TESTING, INPUT_DATA_FILEPATH_TESTING, INPUT_ARRAY_FILEPATH_TESTING, \
    ID_COLUMN_NAME, CLASS_COLUMN_NAME, TESTING_PREDICTION_FILEPATH, TESTING_PREDICTION_PARENT_DIR, \
    CONFUSION_MATRIX_FILENAME_TRAINING, CONFUSION_MATRIX_FILEPATH_VALIDATION, CONFUSION_MATRIX_FILEPATH_TRAINING, \
    PROGRAM_OUTPUT_DIR

from utilities.DebugFlags import LOGISTIC_REGRESSION_DELTA_DEBUG, LOGISTIC_REGRESSION_X_MATRIX_DEBUG, \
    LOGISTIC_REGRESSION_NORMALIZE_W_MATRIX_DEBUG, LOGISTIC_REGRESSION_PROBABILITY_MATRIX_DEBUG, \
    LOGISTIC_REGRESSION_GRADIENT_DESCENT_DEBUG, LOGISTIC_REGRESSION_TRAINING_DEBUG, \
    LOGISTIC_REGRESSION_PREDICTION_DEBUG, LOGISTIC_REGRESSION_ACCURACY_DEBUG, LOGISTIC_REGRESSION_Y_VECTOR_DEBUG, \
    LOGISTIC_REGRESSION_TESTING_ARRAY_DEBUG

import sparse
from math import exp
import numpy as np
from scipy.special import softmax
import math

import time
import glob
import re
import os
from pathlib import Path

# for confusion matrix
import matplotlib.pyplot as plt
import seaborn

class LogisticRegression:
    def __init__(self,
                 input_training_array_da: da.array,
                 input_testing_array_da: da.array,
                 data_parameters: DataParameters,
                 hyperparameters: LogisticRegressionHyperparameters,
                 normalize_W=True,
                 normalize_X=False):
        self.data_parameters = data_parameters
        self.hyperparameters = hyperparameters
        self.normalize_W = normalize_W
        self.normalize_X = normalize_X

        # TODO: add validation
        # remove the index column
        self.input_array_da = input_training_array_da[:, 1:]
        self.testing_array_da, self.testing_id_da = self.create_testing_array(input_testing_array_da)

        # save current iter number (different from num passed into hyperparameters)
        # used in creating the testing prediction file
        self.current_num_iter = None

        # FIXME: should be based on the ENTIRE input data...
        self.class_vector = self.input_array_da[:, -1]

        # number of data instances for training, validation
        self.num_instances_training = None
        self.num_instances_validation = None
        self.set_num_instances_training_validation()

        self.training_array_da, self.validation_array_da = self.create_training_validation_array()

        # keep track of validation accuracy list
        self.validation_accuracy_list = []
        self.validation_accuracy_max = None

        # set execution filepath of X matrix
        self.X_matrix_filepath = None

        # set execution filepath of W matrix
        self.output_dir = f"{OUTPUT_DIR}"
        self.set_parent_dir_W_matrix()
        self.filepath_W_matrix = self.get_filepath_W_matrix(DataOptionEnum.LOAD)
        self.filepath_W_matrix_normalization_string = None

        # number of rows is the number of examples
        # number of columns is the total number of attributes (except for the index column at the beginning and the
        # class column at the end)
        # TODO: different values of m for training, validation...n is the same for both
        self.m_training, self.n = self.training_array_da.shape
        self.m_validation, _ = self.validation_array_da.shape
        self.m_testing, _ = self.testing_array_da.shape

        # do not include the class column
        self.n -= 1
        self.k = len(data_parameters.class_labels_dict)

        ##################################################################

        # create the matrices

        self.delta_matrix = self.create_delta_matrix()

        # print(f"example delta matrix columns")
        # print(f"{self.delta_matrix[:, 0:20].compute()}")

        # TODO: need to create X matrix for both training and validation
        # TODO: also for testing
        self.X_matrix_training = self.create_X_matrix(X_matrix_type=XMatrixType.TRAINING)
        self.X_matrix_validation = self.create_X_matrix(X_matrix_type=XMatrixType.VALIDATION)
        self.X_matrix_testing = self.create_X_matrix(X_matrix_type=XMatrixType.TESTING)

        # TODO: save training/validation Y vector for checking accuracy...
        self.Y_vector_training = self.create_Y_vector(X_matrix_type=XMatrixType.TRAINING)
        self.Y_vector_validation = self.create_Y_vector(X_matrix_type=XMatrixType.VALIDATION)

        self.Y_vector_dict = {XMatrixType.TRAINING: self.Y_vector_training,
                              XMatrixType.VALIDATION: self.Y_vector_validation}

        self.W_matrix = self.create_W_matrix()

    def set_num_instances_training_validation(self):
        hyperparameters = self.hyperparameters
        validation_split = hyperparameters.validation_split

        input_array_num_rows, input_array_num_columns = self.input_array_da.shape
        num_instances_validation = int(validation_split * input_array_num_rows)
        num_instances_training = input_array_num_rows - num_instances_validation

        # FIXME: not ran on EACH run (i.e., not initialized when reading in from file)
        # save for later use
        self.num_instances_training = num_instances_training
        self.num_instances_validation = num_instances_validation

    def generate_training_validation_array(self):
        """
        Simple split

        a - num of training instances
        b - num of validation instances

        first a instances (rows) are for training
        last b instances (rows) are for validation

        :return: training array, validation array
        """

        # hyperparameters = self.hyperparameters
        # validation_split = hyperparameters.validation_split
        #
        # input_array_num_rows, input_array_num_columns = self.input_array_da.shape
        #
        # num_instances_validation = int(validation_split * input_array_num_rows)
        # num_instances_training = input_array_num_rows - num_instances_validation
        #
        # # FIXME: not ran on EACH run (i.e., not initialized when reading in from file)
        # # save for later use
        # self.num_instances_training = num_instances_training
        # self.num_instances_validation = num_instances_validation

        # TODO: need to save...

        num_instances_training = self.num_instances_training
        assert num_instances_training is not None

        training_array = self.input_array_da[0: num_instances_training, :]
        validation_array = self.input_array_da[num_instances_training:, :]

        save_da_array_pickle(training_array, INPUT_ARRAY_FILEPATH_TRAINING)
        save_da_array_pickle(validation_array, INPUT_ARRAY_FILEPATH_VALIDATION)

        return training_array, validation_array

    def create_training_validation_array(self):
        """
        Little tricky since generate_training_validation_array() will return 2 objects instead of the usual one --
        conflicts with fetching only 1 object from a file

        Use isinstance() to check for tuple

        :return:
        """

        training_validation_data = \
            get_data_from_file(DataFileEnum.INPUT_ARRAY_TRAINING,
                               self.generate_training_validation_array,
                               custom_filepath=INPUT_ARRAY_FILEPATH_TRAINING)

        print(f"training, validation: {training_validation_data}")

        if not isinstance(training_validation_data, tuple):
            training_array = training_validation_data
            validation_array = \
                get_data_from_file(DataFileEnum.INPUT_ARRAY_VALIDATION,
                                   self.generate_training_validation_array,
                                   custom_filepath=INPUT_ARRAY_FILEPATH_VALIDATION)

            assert not isinstance(validation_array, tuple)
        else:
            training_array, validation_array = training_validation_data

        return training_array, validation_array

    def create_testing_array(self, input_testing_array_da: da):
        testing_array_da = input_testing_array_da[:, 1:]
        testing_id_da = input_testing_array_da[:, 0]

        return testing_array_da, testing_id_da

    def generate_delta_matrix(self):
        """
        Takes a while to construct the delta matrix because of the for loop...

        :return:
        """

        class_column = self.class_vector
        num_data_rows = len(class_column)

        k, m = self.k, self.m_training
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

    def generate_X_matrix(self, X_matrix_type: XMatrixType = None):
        """
        First column is all 1s (for initial weights w_0 when finding the probability matrix)
        Other columns are just values from the input data (excluding the class column)

        Edit: add option to also create for validation (type)

        :return:
        """

        if LOGISTIC_REGRESSION_X_MATRIX_DEBUG:
            print(f"X matrix type: {X_matrix_type}")

        if X_matrix_type == XMatrixType.TRAINING:
            m = self.m_training
        elif X_matrix_type == XMatrixType.VALIDATION:
            m = self.m_validation
        elif X_matrix_type == XMatrixType.TESTING:
            m = self.m_testing
        else:
            raise ValueError("Invalid X matrix type...")

        n = self.n

        X_matrix = da.zeros((m, n + 1), dtype=int)

        X_matrix[:, 0] = 1

        if LOGISTIC_REGRESSION_X_MATRIX_DEBUG:
            print(f"computing X matrix...")

        # FIXME: expensive operation...
        # TODO: also check for validation type
        if X_matrix_type == XMatrixType.TRAINING:
            X_matrix[:, 1:] = self.training_array_da[:, :-1].compute().todense()
        elif X_matrix_type == XMatrixType.VALIDATION:
            X_matrix[:, 1:] = self.validation_array_da[:, :-1].compute().todense()
        elif X_matrix_type == XMatrixType.TESTING:
            # FIXME: testing data doesn't have a class column...so take all columns
            X_matrix[:, 1:] = self.testing_array_da[:, :].compute().todense()
        else:
            raise ValueError("Invalid X matrix type...")

        # TODO: check for normalization
        if self.normalize_X:
            # normalize each column (along the feature/word instead of each instance)
            X_matrix = da.apply_along_axis(normalize_column_vector,
                                           0,
                                           X_matrix)

        # COO or sparse matrices in general doesn't support item assignment...
        X_matrix = X_matrix.map_blocks(lambda x: sparse.COO(x, fill_value=0))

        if LOGISTIC_REGRESSION_X_MATRIX_DEBUG:
            print(f"input array: {self.training_array_da}")
            print(f"input array: {self.training_array_da.compute()}")

            # print(f"X matrix: intermediate steps...")
            # X_matrix = X_matrix.persist()

        # need to create sub_directories first
        # FIXME: don't include the filename...
        X_matrix_filepath_path = Path(self.X_matrix_filepath)
        X_matrix_parent_path = X_matrix_filepath_path.parent

        create_sub_directories(str(X_matrix_parent_path))

        save_da_array_pickle(X_matrix, self.X_matrix_filepath)

        return X_matrix

    def create_X_matrix(self, X_matrix_type: XMatrixType = None):
        # TODO: edit file path (depending on normalization)
        # TODO: also check for validation type
        if X_matrix_type == XMatrixType.TRAINING:
            data_file_enum = DataFileEnum.X_MATRIX_TRAINING

            if self.normalize_X:
                X_matrix_filepath = X_MATRIX_FILEPATH_X_NORMALIZATION_TRAINING
            else:
                X_matrix_filepath = X_MATRIX_FILEPATH_NO_NORMALIZATION_TRAINING
        elif X_matrix_type == XMatrixType.VALIDATION:
            data_file_enum = DataFileEnum.X_MATRIX_VALIDATION

            if self.normalize_X:
                X_matrix_filepath = X_MATRIX_FILEPATH_X_NORMALIZATION_VALIDATION
            else:
                X_matrix_filepath = X_MATRIX_FILEPATH_NO_NORMALIZATION_VALIDATION
        elif X_matrix_type == XMatrixType.TESTING:
            data_file_enum = DataFileEnum.X_MATRIX_TESTING

            if self.normalize_X:
                X_matrix_filepath = X_MATRIX_FILEPATH_X_NORMALIZATION_TESTING
            else:
                X_matrix_filepath = X_MATRIX_FILEPATH_NO_NORMALIZATION_TESTING
        else:
            raise ValueError("Invalid X matrix type...")

        self.X_matrix_filepath = X_matrix_filepath

        # TODO: add kwargs for passing optional args to generate_X_matrix()
        X_matrix = get_data_from_file(data_file_enum,
                                      self.generate_X_matrix,
                                      custom_filepath=X_matrix_filepath,
                                      X_matrix_type=X_matrix_type)

        if LOGISTIC_REGRESSION_X_MATRIX_DEBUG:
            # check stats of X matrix...
            print(f"compute max X matrix...")
            print(f"{da.max(X_matrix).compute()}")

        return X_matrix

    def create_Y_vector(self, X_matrix_type: XMatrixType = None):
        num_instances_training = self.num_instances_training
        num_instances_validation = self.num_instances_validation

        assert num_instances_training is not None

        if X_matrix_type == XMatrixType.TRAINING:
            Y_vector = self.class_vector[0: num_instances_training]
        elif X_matrix_type == XMatrixType.VALIDATION:
            Y_vector = self.class_vector[num_instances_training:]
        else:
            raise ValueError("Invalid X matrix type...")

        if LOGISTIC_REGRESSION_Y_VECTOR_DEBUG:
            print(f"-----------------------------------------------------")
            print(f"num instances training: {num_instances_training}")
            print(f"x matrix type: {X_matrix_type}")
            print(f"y vector shape: {Y_vector.shape}")
            print(f"class_vector shape: {self.class_vector.shape}")
            print(f"-----------------------------------------------------")

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

    def get_W_matrix_hyperparameters_dir(self):
        hyperparameters = self.hyperparameters

        training_eta = hyperparameters.learning_rate
        training_lambda = hyperparameters.penalty_term

        return f"eta={training_eta},lambda={training_lambda}"

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

    def set_parent_dir_W_matrix(self):
        output_dir = self.output_dir
        W_matrix_sub_dir = self.get_W_matrix_sub_dir()
        W_matrix_hyperparameters_dir = self.get_W_matrix_hyperparameters_dir()
        W_matrix_parent_dir = os.path.join(output_dir, W_MATRIX_TOP_DIR, W_matrix_sub_dir,
                                           W_matrix_hyperparameters_dir)

        self.W_matrix_parent_dir = W_matrix_parent_dir

    def get_filepath_W_matrix(self, data_option: DataOptionEnum, current_iter=None):
        """
        format of filepath

        w_matrix - top level dir
        |- no_normalization - sub dir
        |- w_normalization - sub dir
        |- x_normalization - sub dir
        |- w_x_normalization - sub dir

        Under the sub dir, there will be another dir that specifies the eta and lambda values used for the training

        e.g., "eta=0.01,lambda=0.001"

        file name
        w_matrix-<iter_num>.pkl

        e.g., w_matrix-10.pkl (after a total of 10 iterations, the corresponding W matrix was saved)

        will continue on from the most recent (i.e., file with the highest iter_num) on next execution

        iter_num - the collective amount of iterations (gradient descent) trained on the W matrix

        :param current_iter:
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

        # output_dir = self.output_dir
        # W_matrix_sub_dir = self.get_W_matrix_sub_dir()
        # W_matrix_hyperparameters_dir = self.get_W_matrix_hyperparameters_dir()
        # W_matrix_parent_dir = os.path.join(output_dir, W_MATRIX_TOP_DIR, W_matrix_sub_dir, W_matrix_hyperparameters_dir)

        W_matrix_parent_dir = self.W_matrix_parent_dir

        print(f"W matrix parent dir: {W_matrix_parent_dir}")

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
            # TODO: instead of using num_iter, pass in iteration number (may save MULTIPLE times in the same run...)
            # next_iter_num = self.hyperparameters.num_iter

            assert current_iter is not None
            next_iter_num = current_iter

            next_iter_num += prev_iter_num
            iter_string += f"-{next_iter_num}"

            self.current_num_iter = next_iter_num
        elif data_option == DataOptionEnum.LOAD:
            iter_string += f"-{prev_iter_num}"

            self.current_num_iter = prev_iter_num
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

    def compute_probability_matrix(self, X_matrix_type: XMatrixType = None):
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

        # get X matrix depending on type
        if X_matrix_type == XMatrixType.TRAINING:
            X_matrix = self.X_matrix_training
        elif X_matrix_type == XMatrixType.VALIDATION:
            X_matrix = self.X_matrix_validation
        elif X_matrix_type == XMatrixType.TESTING:
            X_matrix = self.X_matrix_testing
        else:
            raise ValueError("Invalid X matrix type...")

        # compute un-normalized probability matrix
        probability_Y_given_W_X_matrix_no_exp = da.dot(self.W_matrix,
                                                       da.transpose(X_matrix))

        if LOGISTIC_REGRESSION_PROBABILITY_MATRIX_DEBUG:
            print(f"computing max...")
            print(f"max no exp: {da.max(probability_Y_given_W_X_matrix_no_exp).compute()}")

            print(f"type of prob: {type(probability_Y_given_W_X_matrix_no_exp)}")
            print(f"prob without exp: {probability_Y_given_W_X_matrix_no_exp[0][0:6].compute()}")

            # print(f"BEFORE EXP compute probability matrix...")
            # print(f"{self.W_matrix.compute()}")
            # print(f"{self.X_matrix_training.compute()}")
            # probability_Y_given_W_X_matrix_no_exp.compute()

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

    def get_prediction_vector(self, X_matrix_type: XMatrixType = None):
        """
        Should be a (, m) vector containing the predictions for each data instance (m total data instances,
        depending on which X matrix is used)

        :param X_matrix_type:
        :return:
        """

        probability_matrix = self.compute_probability_matrix(X_matrix_type=X_matrix_type)

        # get argmax along the columns
        prediction_vector_argmax = da.argmax(probability_matrix, axis=0)

        # add 1 since indexing starts at 0 and classes start at 1
        prediction_vector_argmax = prediction_vector_argmax + 1
        prediction_vector_argmax = prediction_vector_argmax.persist()

        return prediction_vector_argmax

    def get_accuracy(self, X_matrix_type: XMatrixType = None):
        prediction_vector = self.get_prediction_vector(X_matrix_type=X_matrix_type)
        Y_vector = self.Y_vector_dict[X_matrix_type]

        if LOGISTIC_REGRESSION_ACCURACY_DEBUG:
            print(f"x matrix type: {X_matrix_type}")
            print(f"prediction vector shape: {prediction_vector.shape}")
            print(f"Y_vector shape: {Y_vector.shape}")

        boolean_vector = da.where(prediction_vector == Y_vector, 1, 0)
        accuracy = da.sum(boolean_vector) / len(boolean_vector)

        # FIXME: need to compute...
        accuracy = accuracy.compute()

        return accuracy

    def compute_gradient_descent_step(self):
        hyperparameters = self.hyperparameters
        learning_rate = hyperparameters.learning_rate
        penalty_term = hyperparameters.penalty_term

        if LOGISTIC_REGRESSION_GRADIENT_DESCENT_DEBUG:
            print(f"W matrix: {self.W_matrix}")
            print(f"delta matrix: {self.delta_matrix}")
            print(f"X: matrix {self.X_matrix_training}")

        probability_Y_given_W_X_matrix = self.compute_probability_matrix(X_matrix_type=XMatrixType.TRAINING)

        intermediate_W_matrix = learning_rate * \
                                (da.dot((self.delta_matrix - probability_Y_given_W_X_matrix), self.X_matrix_training) -
                                 penalty_term * self.W_matrix)

        if LOGISTIC_REGRESSION_GRADIENT_DESCENT_DEBUG:
            print(f"shape intermediate: {intermediate_W_matrix.shape}")
            print(f"delta - P: {da.max(self.delta_matrix - probability_Y_given_W_X_matrix).compute()}")
            print(f"(delta - P)X: "
                  f"{da.max(da.dot((self.delta_matrix - probability_Y_given_W_X_matrix), self.X_matrix_training)).compute()}")
            print(f"lambda W: {da.max(penalty_term * self.W_matrix).compute()}")
            print(f"intermediate W: {da.max(intermediate_W_matrix).compute()}")

            print(f"intermediate W: {intermediate_W_matrix.compute()}")
            print(f"intermediate W dot: "
                  f"{da.dot((self.delta_matrix - probability_Y_given_W_X_matrix), self.X_matrix_training).compute()}")

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
        hyperparameters = self.hyperparameters
        num_iter_print = hyperparameters.num_iter_print
        num_iter_save = hyperparameters.num_iter_save
        num_iter_validation = hyperparameters.num_iter_validation
        validation_accuracy_diff_cutoff = hyperparameters.validation_accuracy_diff_cutoff

        print(f"Starting training...")

        current_num_iter_execution = self.hyperparameters.num_iter
        num_iter = current_num_iter_execution

        if current_num_iter_execution == 0:
            print(f"No training done")
            # return

        for i in range(current_num_iter_execution):
            self.compute_gradient_descent_step()

            if LOGISTIC_REGRESSION_TRAINING_DEBUG:
                print(f"Iter {i} complete")

            if (i + 1) % num_iter_print == 0:
                print(f"-----------------------------------")
                print(f"Finished {i + 1} iterations")
                print(f"-----------------------------------")

            if (i + 1) % num_iter_save == 0:
                print(f"-----------------------------------")
                print(f"Saving after {i + 1} iterations...")
                save_da_array_pickle(self.W_matrix, self.get_filepath_W_matrix(DataOptionEnum.SAVE, num_iter_save))
                print(f"Saving complete")
                print(f"-----------------------------------")

            # Simple validation
            if (i + 1) % num_iter_validation == 0:
                print(f"-----------------------------------")
                print(f"Checking accuracy of training...")
                training_accuracy = self.get_accuracy(X_matrix_type=XMatrixType.TRAINING)
                print(f"training accuracy: {training_accuracy}")
                print(f"-----------------------------------")

                print(f"-----------------------------------")
                print(f"Checking validation...")
                validation_accuracy = self.get_accuracy(X_matrix_type=XMatrixType.VALIDATION)
                print(f"validation accuracy: {validation_accuracy}")
                print(f"-----------------------------------")

                current_validation_accuracy_max = self.validation_accuracy_max

                if current_validation_accuracy_max is None:
                    self.validation_accuracy_max = validation_accuracy
                elif validation_accuracy > current_validation_accuracy_max:
                    self.validation_accuracy_max = validation_accuracy
                else:
                    # stop training if validation worsens by a certain margin from the best validation accuracy
                    if current_validation_accuracy_max - validation_accuracy > validation_accuracy_diff_cutoff:
                        num_iter = i + 1
                        break

                # maybe to plot later
                self.validation_accuracy_list.append(validation_accuracy)

                # # validation list is empty
                # if not self.validation_accuracy_list:
                #     pass

        if LOGISTIC_REGRESSION_TRAINING_DEBUG:
            # finish W matrix
            print(f"saving W matrix...")
            start_time = time.time()

        # self.W_matrix = self.W_matrix.compute()

        # complete validation for 0 iterations (DEBUG stuff)
        if current_num_iter_execution == 0:
            print(f"-----------------------------------")
            print(f"Checking accuracy of training...")
            training_accuracy = self.get_accuracy(X_matrix_type=XMatrixType.TRAINING)
            print(f"training accuracy: {training_accuracy}")
            print(f"-----------------------------------")

            print(f"-----------------------------------")
            print(f"Checking validation...")
            validation_accuracy = self.get_accuracy(X_matrix_type=XMatrixType.VALIDATION)
            print(f"validation accuracy: {validation_accuracy}")
            print(f"-----------------------------------")

        # TODO: save W matrix into file
        # TODO: find previous iteration num among files...should work if files are deleted
        num_iter_remainder = num_iter % num_iter_save
        if num_iter_remainder != 0:
            print(f"-----------------------------------")
            print(f"Saving {num_iter_remainder} remainder iterations...")
            save_da_array_pickle(self.W_matrix, self.get_filepath_W_matrix(DataOptionEnum.SAVE,
                                                                           num_iter_remainder))
            print(f"Saving complete")
            print(f"-----------------------------------")

        if LOGISTIC_REGRESSION_TRAINING_DEBUG:
            end_time = time.time()
            print(f"W matrix time: {end_time - start_time}")

        print(f"-----------------------------------")
        print(f"Training complete")
        print(f"-----------------------------------")

    def create_testing_file(self, custom_num_iter=None):
        print(f"Getting testing prediction vector...")

        # TODO: save the hyperparameters and iter num for each prediction...
        hyperparameters = self.hyperparameters
        eta_training = hyperparameters.learning_rate
        lambda_training = hyperparameters.penalty_term
        current_num_iter = self.current_num_iter

        if custom_num_iter is not None:
            # need to rewrite the num iter saved in filename
            current_num_iter = custom_num_iter

            # TODO: reload W matrix with corresponding custom_num_iter
            W_matrix_parent_dir = self.W_matrix_parent_dir

            # get filename
            W_matrix_filename = f"{W_MATRIX_FILENAME_WITHOUT_EXTENSION}-{custom_num_iter}" \
                                f"{W_MATRIX_FILENAME_EXTENSION}"

            print(f"W matrix parent dir: {W_matrix_parent_dir}")
            print(f"W matrix filename: {W_matrix_filename}")

            # create path for W matrix file
            W_matrix_filepath = os.path.join(W_matrix_parent_dir, W_matrix_filename)

            self.W_matrix = load_da_array_pickle(W_matrix_filepath)

            # check validation AFTER LOADING in the W matrix
            print(f"-----------------------------------")
            print(f"Checking validation...")
            validation_accuracy = self.get_accuracy(X_matrix_type=XMatrixType.VALIDATION)
            print(f"validation accuracy: {validation_accuracy}")
            print(f"-----------------------------------")

        testing_prediction_vector = self.get_prediction_vector(X_matrix_type=XMatrixType.TESTING)

        print(f"Saving testing prediction into file...")

        print(f"Converting ID column to dense...")
        print(f"id column: {self.testing_id_da}")

        self.testing_id_da = self.testing_id_da.compute().todense()

        print(f"Densification complete...")

        testing_dict = {ID_COLUMN_NAME: self.testing_id_da,
                        CLASS_COLUMN_NAME: testing_prediction_vector}
        testing_df = pd.DataFrame(testing_dict)

        # need to create directory first
        # FIXME: only parent dir, not entire filepath...
        create_sub_directories(TESTING_PREDICTION_PARENT_DIR)

        testing_prediction_filename = \
            f"testing_prediction_eta={eta_training}-lambda={lambda_training}-iter_num={current_num_iter}.csv"

        testing_prediction_filepath = os.path.join(TESTING_PREDICTION_PARENT_DIR, testing_prediction_filename)

        testing_df.to_csv(testing_prediction_filepath, index=False)

        print(f"Saving complete")
        print(f"save location: {testing_prediction_filepath}")

        print(f"-----------------------------------")

    def set_initial_parameters(self):
        pass

    def get_word_rank_list(self):
        """
        For information gain, entropy was used where each p_i used is just the probability for each document
        classification (class) for i = 1, 2,..., 20

        :return:
        """

        # find entropy for each word
        probability_matrix = self.compute_probability_matrix(X_matrix_type=XMatrixType.TESTING)

        print(f"probability shape: {probability_matrix.shape}")
        first_row = probability_matrix[0, :].compute()
        print(f"probability: {first_row}")
        print(f"probability check: {-sum(p * math.log2(p) for p in first_row)}")

        # entropy_word_vector = da.apply_along_axis(calculate_entropy_column_vector,
        #                                           0,
        #                                           probability_matrix)
        entropy_word_vector = da.apply_along_axis(calculate_gini_index_column_vector,
                                                  0,
                                                  probability_matrix)

        print(f"shape: {entropy_word_vector.shape}")

        dtype = [("index", int), ("entropy", float)]
        # new_entropy_word_vector = np.zeros_like(entropy_word_vector, dtype=dtype)
        # entropy_word_vector.astype(dtype)

        # entropy_index_list = []
        #
        # print(f"BEFORE")
        #
        # for i, entropy in enumerate(entropy_word_vector):
        #     # new_entropy_word_vector[i] = (i, entropy)
        #     entropy_index_list.append((i, entropy))
        #
        # print(f"AFTER")
        #
        # # FIXME: long time...
        # new_entropy_word_vector = da.asarray(entropy_index_list, dtype=dtype)

        print(f"SORT BEFORE")
        arg_sort = da.argtopk(entropy_word_vector, 100).compute()
        entropy_sort = da.topk(entropy_word_vector, 100).compute()
        # entropy_word_vector_sorted = np.sort(new_entropy_word_vector, order="entropy")

        print(f"SORT AFTER")

        print(f"arg sort: {arg_sort}")
        print(f"entropy/gini index sort: {entropy_sort}")

        # return top 100 words
        return arg_sort, entropy_sort

    def generate_confusion_matrix(self):
        """
        Create confusion matrix for both training and validation data sets

        :return:
        """

        # get actual class vector for both training, validation
        class_vector_training_actual = self.Y_vector_training
        class_vector_validation_actual = self.Y_vector_validation

        # prediction for training
        class_vector_training_prediction = self.get_prediction_vector(X_matrix_type=XMatrixType.TRAINING)

        # prediction for validation
        class_vector_validation_prediction = self.get_prediction_vector(X_matrix_type=XMatrixType.VALIDATION)

        # get counts for the different classes
        class_labels_dict = self.data_parameters.class_labels_dict
        num_classes = len(class_labels_dict)

        # confusion matrix for training
        confusion_matrix_da_training = da.zeros((num_classes, num_classes))

        # columns are ground truth (actual) and rows are the observed (prediction) for the corresponding actual class
        # find the counts for each class along the columns
        for i in range(num_classes):
            confusion_matrix_da_training[:, i] = \
                self.get_class_count_column_vector(confusion_matrix_da_training[:, i],
                                                   class_vector_type=ClassVectorType.TRAINING,
                                                   class_index=i,
                                                   num_classes=num_classes)

        # confusion matrix for validation
        confusion_matrix_da_validation = da.zeros((num_classes, num_classes))

        # columns are ground truth (actual) and rows are the observed (prediction) for the corresponding actual class
        # find the counts for each class along the columns
        for i in range(num_classes):
            confusion_matrix_da_validation[:, i] = \
                self.get_class_count_column_vector(confusion_matrix_da_validation[:, i],
                                                   class_vector_type=ClassVectorType.VALIDATION,
                                                   class_index=i,
                                                   num_classes=num_classes)

        # FIXME: no way to access index with apply_along_axis...
        # da.where(class_vector_training_prediction == )
        # confusion_matrix_da_training = da.apply_along_axis(self.get_class_count_column_vector,
        #                                           0,
        #                                           confusion_matrix_da_training,
        #                                                    class_vector_type=ClassVectorType.TRAINING, class_index)

        hyperparameters = self.hyperparameters
        eta_training = hyperparameters.learning_rate
        lambda_training = hyperparameters.penalty_term
        current_num_iter = self.current_num_iter

        confusion_matrix_filename_training = \
            f"confusion_matrix_training_eta={eta_training}-lambda={lambda_training}-iter_num={current_num_iter}.pkl"

        confusion_matrix_filepath_training = os.path.join(PROGRAM_OUTPUT_DIR, confusion_matrix_filename_training)

        confusion_matrix_filename_validation = \
            f"confusion_matrix_validation_eta={eta_training}-lambda={lambda_training}-iter_num={current_num_iter}.pkl"

        confusion_matrix_filepath_validation = os.path.join(PROGRAM_OUTPUT_DIR, confusion_matrix_filename_validation)

        # save to file
        save_da_array_pickle(confusion_matrix_da_training, confusion_matrix_filepath_training)
        save_da_array_pickle(confusion_matrix_da_validation, confusion_matrix_filepath_validation)

        return confusion_matrix_da_training.compute(), confusion_matrix_da_validation.compute()

    def create_confusion_matrix(self):
        """
        Little tricky since generate_training_validation_array() will return 2 objects instead of the usual one --
        conflicts with fetching only 1 object from a file

        Use isinstance() to check for tuple

        :return:
        """

        hyperparameters = self.hyperparameters
        eta_training = hyperparameters.learning_rate
        lambda_training = hyperparameters.penalty_term
        current_num_iter = self.current_num_iter

        confusion_matrix_filename_training = \
            f"confusion_matrix_training_eta={eta_training}-lambda={lambda_training}-iter_num={current_num_iter}.pkl"

        confusion_matrix_filepath_training = os.path.join(PROGRAM_OUTPUT_DIR, confusion_matrix_filename_training)

        confusion_matrix_filename_validation = \
            f"confusion_matrix_validation_eta={eta_training}-lambda={lambda_training}-iter_num={current_num_iter}.pkl"

        confusion_matrix_filepath_validation = os.path.join(PROGRAM_OUTPUT_DIR, confusion_matrix_filename_validation)

        confusion_matrix_da_training_validation = \
            get_data_from_file(DataFileEnum.CONFUSION_MATRIX_TRAINING,
                               self.generate_confusion_matrix,
                               custom_filepath=confusion_matrix_filepath_training)

        print(f"training, validation: {confusion_matrix_da_training_validation}")

        if not isinstance(confusion_matrix_da_training_validation, tuple):
            confusion_matrix_da_training = confusion_matrix_da_training_validation
            confusion_matrix_da_validation = \
                get_data_from_file(DataFileEnum.CONFUSION_MATRIX_VALIDATION,
                                   self.generate_confusion_matrix,
                                   custom_filepath=confusion_matrix_filepath_validation)

            assert not isinstance(confusion_matrix_da_validation, tuple)
        else:
            confusion_matrix_da_training, confusion_matrix_da_validation = confusion_matrix_da_training_validation

        return confusion_matrix_da_training, confusion_matrix_da_validation

    def get_class_count_column_vector(self, count_vector, **kwargs):
        args = kwargs
        class_vector_type = args["class_vector_type"]
        class_index = args["class_index"]
        num_classes = args["num_classes"]

        if class_vector_type == ClassVectorType.TRAINING:
            class_vector_actual = self.Y_vector_training
            class_vector_prediction = self.get_prediction_vector(X_matrix_type=XMatrixType.TRAINING)
        elif class_vector_type == ClassVectorType.VALIDATION:
            class_vector_actual = self.Y_vector_validation
            class_vector_prediction = self.get_prediction_vector(X_matrix_type=XMatrixType.VALIDATION)
        else:
            raise ValueError("Invalid class vector type...")

        # FIXME: maybe better to do with dask dataframe...
        # TODO: need to convert sparse to dense...only for actual class vector (prediction vector is already dense)
        class_vector_actual = class_vector_actual.compute().todense()
        class_vector_prediction = class_vector_prediction

        # FIXME: use raw DataFrame instead of dask version for indexing issues? (chunk size)
        # use default indexing from 0
        class_vector_prediction_ddf = df.from_dask_array(class_vector_prediction).compute()

        # get value counts (need to add one for classes)
        counts_series = class_vector_prediction_ddf[class_vector_actual == (class_index + 1)].value_counts()

        # get the indices and values
        counts_index = counts_series.index
        # need to subtract one for indexing back to 0...
        counts_index -= 1

        # FIXME: invalid Boolean indexing (maybe due to different chunk sizes?)
        counts_index = counts_index
        counts_values = counts_series.values

        # counts_index = counts_index.compute()
        # counts_values = counts_series.values.compute()

        counts_array_da = da.zeros((num_classes, ))
        counts_array_da[counts_index] = counts_values

        return counts_array_da

    def plot_confusion_matrix(self):
        """
        Adapted from the same function for Naive Bayes

        :return:
        """

        confusion_matrix_da_training, confusion_matrix_da_validation = self.create_confusion_matrix()

        print(f"confusion_matrix_training: {confusion_matrix_da_training}")
        print(f"confusion_matrix_validation: {confusion_matrix_da_validation}")

        hyperparameters = self.hyperparameters
        eta_training = hyperparameters.learning_rate
        lambda_training = hyperparameters.penalty_term
        current_num_iter = self.current_num_iter

        class_labels_dict = self.data_parameters.class_labels_dict
        news_groups = [news + f" ({index})" for index, news in class_labels_dict.items()]

        # save the plots
        confusion_matrix_plot_filename_training = \
            f"confusion_matrix_plot_training_eta={eta_training}-lambda={lambda_training}-iter_num" \
            f"={current_num_iter}.png"

        confusion_matrix_plot_filepath_training = \
            os.path.join(PROGRAM_OUTPUT_DIR, confusion_matrix_plot_filename_training)

        confusion_matrix_plot_filename_validation = \
            f"confusion_matrix_plot_validation_eta={eta_training}-lambda={lambda_training}-iter_num" \
            f"={current_num_iter}.png"

        confusion_matrix_plot_filepath_validation = \
            os.path.join(PROGRAM_OUTPUT_DIR, confusion_matrix_plot_filename_validation)

        f = plt.figure(1, figsize=(16, 10))

        plt.rc('axes', titlesize=20)
        plt.rc('axes', labelsize=15)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.rc('legend', fontsize=12)
        # fig = plt.figure(figsize=(16, 10))
        ax = f.add_subplot(1, 1, 1)
        ax.set_title(f"Confusion Matrix Training "
                     f"(eta={eta_training},lambda={lambda_training},num iter={current_num_iter})")

        seaborn.heatmap(confusion_matrix_da_training,
                        annot=True, fmt=".0f", xticklabels=news_groups, yticklabels=news_groups)  # plot
        ax.set_xticklabels(news_groups, rotation=60)

        f.savefig(confusion_matrix_plot_filepath_training, format='png')
        f.show()

        g = plt.figure(2, figsize=(16, 10))

        plt.rc('axes', titlesize=20)
        plt.rc('axes', labelsize=15)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.rc('legend', fontsize=12)
        # fig = plt.figure(figsize=(16, 10))
        ax = g.add_subplot(1, 1, 1)
        ax.set_title(f"Confusion Matrix Validation "
                     f"(eta={eta_training},lambda={lambda_training},num iter={current_num_iter})")

        seaborn.heatmap(confusion_matrix_da_validation,
                        annot=True, fmt=".0f", xticklabels=news_groups, yticklabels=news_groups)  # plot
        ax.set_xticklabels(news_groups, rotation=60)

        g.savefig(confusion_matrix_plot_filepath_validation, format='png')
        g.show()


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
