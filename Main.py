from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# parameter imports
from parameters.DataParameters import \
    DataParameters

from parameters.Hyperparameters import \
    LogisticRegressionHyperparameters

# logistic regression imports
from logistic_regression.LogisticRegression import \
    LogisticRegression

# utilities/helper imports
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from utilities.ParseUtilities import \
    parse_class_labels, get_training_data, generate_training_data, get_data_from_file

from utilities.DataFile import DataFileEnum

from Constants import \
    INPUT_ARRAY_FILEPATH_TRAINING, CLASS_LABELS_FILEPATH

import time

# Global variables
PROGRESS_BAR_SET = True
MAIN_DEBUG = False
MAIN_PRINT = True


def logistic_regression_training():
    test_logistic_regression.complete_training()


if __name__ == "__main__":
    # prepare progress bar
    if PROGRESS_BAR_SET:
        pbar = ProgressBar()
        pbar.register()  # global registration

    ######################################

    class_labels_dict = parse_class_labels(CLASS_LABELS_FILEPATH)

    print(f"class labels dict: {class_labels_dict}")

    data_parameters = DataParameters(class_labels_dict)

    # tested learning rate = 0.01 and penalty term = 0.01 and the weights exploded...NaN popped up within a few
    # iterations
    hyperparameters = LogisticRegressionHyperparameters(0.01, 0.001, 40, 20, 20, 50)

    # sparse_da_training = get_training_data()
    sparse_da_training = get_data_from_file(DataFileEnum.INPUT_DATA_TRAINING, generate_training_data)

    test_logistic_regression = LogisticRegression(sparse_da_training, data_parameters, hyperparameters,
                                                  normalize_W=False, normalize_X=True)

    start_time = time.time()

    if False:
        # introduce parallelism with multiple workers/threads...slower though and can't see the progress bar
        local_cluster = LocalCluster()

        with Client(local_cluster) as client:
            print(f"local cluster: {local_cluster}")
            print(f"client: {client}")

            logistic_regression_training()

            # gather_W_matrix = client.gather(test_logistic_regression.get_W_matrix())
            # test_logistic_regression.set_W_matrix(gather_W_matrix)
    else:
        logistic_regression_training()

    end_time = time.time()

    print(f"total time: {end_time - start_time}")

    ##############################################################

    # test prediction

    start_time = time.time()

    test_prediction = test_logistic_regression.get_prediction(sparse_da_training[2, :])

    print(f"test prediction: {test_prediction}")
    print(f"actual: {sparse_da_training[0, -1].compute()}")

    end_time = time.time()

    print(f"total prediction time: {end_time - start_time}")
