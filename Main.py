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

from utilities.ParseUtilities import \
    parse_class_labels, get_training_data, generate_training_data, get_data_from_file

from utilities.DataFile import DataFileEnum

from Constants import \
    OUTPUT_FILEPATH_TRAINING, CLASS_LABELS_FILEPATH

# Global variables
MAIN_DEBUG = False
MAIN_PRINT = True


if __name__ == "__main__":
    # prepare progress bar
    pbar = ProgressBar()
    pbar.register()  # global registration

    class_labels_dict = parse_class_labels(CLASS_LABELS_FILEPATH)

    print(f"class labels dict: {class_labels_dict}")

    data_parameters = DataParameters(class_labels_dict)

    # tested learning rate = 0.01 and penalty term = 0.01 and the weights exploded...NaN popped up within a few
    # iterations
    hyperparameters = LogisticRegressionHyperparameters(0.001, 0.01, 50)

    # sparse_da_training = get_training_data()
    sparse_da_training = get_data_from_file(DataFileEnum.OUTPUT_ARRAY_TRAINING,
                                            generate_training_data)

    test_logistic_regression = LogisticRegression(sparse_da_training,
                                                  data_parameters,
                                                  hyperparameters)

    test_logistic_regression.complete_training()

    ##############################################################

    # test prediction

    test_prediction = test_logistic_regression.get_prediction(sparse_da_training[0, :])

    print(f"test prediction: {test_prediction}")
    print(f"actual: {sparse_da_training[0, -1].compute()}")
