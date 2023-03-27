from utilities.DataFile import WMatrixOptionEnum, XMatrixOptionEnum
from utilities.FileSystemUtilities import create_sub_directories

import os

CHUNK_SIZE = 100

# column names in TESTING csv prediction file
ID_COLUMN_NAME = f"id"
CLASS_COLUMN_NAME = f"class"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_SEP = os.path.sep
DIR_UP = f".."

# TODO: added the very top level for specifying training/validation split
# FIXME: too many dependencies to change...hard code "validation_split=0.2"
OUTPUT_DIR = os.path.join(f"resources", f"validation_split=0.2")
create_sub_directories(OUTPUT_DIR)

INPUT_DATA_FILEPATH_TRAINING = f"cs429529-project-2-topic-categorization{PATH_SEP}training.csv"
INPUT_DATA_FILEPATH_TESTING = f"cs429529-project-2-topic-categorization{PATH_SEP}testing.csv"

INPUT_ARRAY_FILENAME_ENTIRE_DATA = f"entire_input_data.pkl"
INPUT_ARRAY_FILEPATH_ENTIRE_DATA = os.path.join(OUTPUT_DIR, INPUT_ARRAY_FILENAME_ENTIRE_DATA)

INPUT_ARRAY_FILENAME_TRAINING = f"training_array.pkl"
# OUTPUT_FILEPATH_TRAINING = f"{OUTPUT_DIR}{OUTPUT_FILENAME_TRAINING}"
INPUT_ARRAY_FILEPATH_TRAINING = os.path.join(OUTPUT_DIR, INPUT_ARRAY_FILENAME_TRAINING)

INPUT_ARRAY_FILENAME_VALIDATION = f"validation_array.pkl"
INPUT_ARRAY_FILEPATH_VALIDATION = os.path.join(OUTPUT_DIR, INPUT_ARRAY_FILENAME_VALIDATION)

INPUT_ARRAY_FILENAME_TESTING = f"testing_array.pkl"
INPUT_ARRAY_FILEPATH_TESTING = os.path.join(OUTPUT_DIR, INPUT_ARRAY_FILENAME_TESTING)

# OUTPUT_CHUNK = (1, 61190)
# OUTPUT_SHAPE = (12000, 61190)

CLASS_LABELS_FILENAME = f"newsgrouplabels.txt"
CLASS_LABELS_FILEPATH = f"cs429529-project-2-topic-categorization{PATH_SEP}{CLASS_LABELS_FILENAME}"

DELTA_MATRIX_FILENAME = f"delta_matrix.pkl"
DELTA_MATRIX_FILEPATH = f"{OUTPUT_DIR}{PATH_SEP}{DELTA_MATRIX_FILENAME}"

X_MATRIX_FILENAME = f"x_matrix.pkl"
# X_MATRIX_FILEPATH_OLD = f"{OUTPUT_DIR}{PATH_SEP}{X_MATRIX_FILENAME}"
X_MATRIX_TOP_DIR = f"x_matrix"
X_MATRIX_TOP_DIR_TRAINING = f"x_matrix_training"
X_MATRIX_TOP_DIR_VALIDATION = f"x_matrix_validation"
X_MATRIX_TOP_DIR_TESTING = f"x_matrix_testing"
X_MATRIX_SUB_DIR_DICT = {XMatrixOptionEnum.NO_NORMALIZATION: f"no_normalization",
                         XMatrixOptionEnum.X_NORMALIZATION: f"x_normalization"}

# Training
X_MATRIX_PARENT_DIR_TRAINING = f"{OUTPUT_DIR}{PATH_SEP}{X_MATRIX_TOP_DIR}{PATH_SEP}" \
                               f"{X_MATRIX_TOP_DIR_TRAINING}{PATH_SEP}"
X_MATRIX_FILEPATH_NO_NORMALIZATION_TRAINING = \
    f"{X_MATRIX_PARENT_DIR_TRAINING}{X_MATRIX_SUB_DIR_DICT[XMatrixOptionEnum.NO_NORMALIZATION]}{PATH_SEP}" \
    f"{X_MATRIX_FILENAME}"
X_MATRIX_FILEPATH_X_NORMALIZATION_TRAINING = \
    f"{X_MATRIX_PARENT_DIR_TRAINING}{X_MATRIX_SUB_DIR_DICT[XMatrixOptionEnum.X_NORMALIZATION]}{PATH_SEP}" \
    f"{X_MATRIX_FILENAME}"

# Validation
X_MATRIX_PARENT_DIR_VALIDATION = f"{OUTPUT_DIR}{PATH_SEP}{X_MATRIX_TOP_DIR}{PATH_SEP}" \
                                 f"{X_MATRIX_TOP_DIR_VALIDATION}{PATH_SEP}"
X_MATRIX_FILEPATH_NO_NORMALIZATION_VALIDATION = \
    f"{X_MATRIX_PARENT_DIR_VALIDATION}{X_MATRIX_SUB_DIR_DICT[XMatrixOptionEnum.NO_NORMALIZATION]}{PATH_SEP}" \
    f"{X_MATRIX_FILENAME}"
X_MATRIX_FILEPATH_X_NORMALIZATION_VALIDATION = \
    f"{X_MATRIX_PARENT_DIR_VALIDATION}{X_MATRIX_SUB_DIR_DICT[XMatrixOptionEnum.X_NORMALIZATION]}{PATH_SEP}" \
    f"{X_MATRIX_FILENAME}"

# TODO: Testing for X matrix
# Testing
X_MATRIX_PARENT_DIR_TESTING = f"{OUTPUT_DIR}{PATH_SEP}{X_MATRIX_TOP_DIR}{PATH_SEP}" \
                                 f"{X_MATRIX_TOP_DIR_TESTING}{PATH_SEP}"
X_MATRIX_FILEPATH_NO_NORMALIZATION_TESTING = \
    f"{X_MATRIX_PARENT_DIR_TESTING}{X_MATRIX_SUB_DIR_DICT[XMatrixOptionEnum.NO_NORMALIZATION]}{PATH_SEP}" \
    f"{X_MATRIX_FILENAME}"
X_MATRIX_FILEPATH_X_NORMALIZATION_TESTING = \
    f"{X_MATRIX_PARENT_DIR_TESTING}{X_MATRIX_SUB_DIR_DICT[XMatrixOptionEnum.X_NORMALIZATION]}{PATH_SEP}" \
    f"{X_MATRIX_FILENAME}"

W_MATRIX_FILENAME_WITHOUT_EXTENSION = f"w_matrix"
W_MATRIX_FILENAME_EXTENSION = f".pkl"
# W_MATRIX_FILEPATH = f"{OUTPUT_DIR}{W_MATRIX_FILENAME}"
W_MATRIX_TOP_DIR = f"w_matrix"
W_MATRIX_SUB_DIR_DICT = {WMatrixOptionEnum.NO_NORMALIZATION: f"no_normalization",
                         WMatrixOptionEnum.W_NORMALIZATION: "w_normalization",
                         WMatrixOptionEnum.X_NORMALIZATION: f"x_normalization",
                         WMatrixOptionEnum.W_X_NORMALIZATION: f"w_x_normalization"}

# Testing predictions output file
TESTING_PREDICTION_FILENAME = f"testing_prediction.csv"
TESTING_PREDICTION_TOP_DIR = f"testing_prediction"
TESTING_PREDICTION_PARENT_DIR = f"{OUTPUT_DIR}{PATH_SEP}{TESTING_PREDICTION_TOP_DIR}{PATH_SEP}"
TESTING_PREDICTION_FILEPATH = \
    f"{TESTING_PREDICTION_PARENT_DIR}{TESTING_PREDICTION_FILENAME}"

# Confusion matrix
PROGRAM_OUTPUT_DIR = f"program_output"

CONFUSION_MATRIX_FILENAME_TRAINING = f"confusion_matrix_lg_training.pkl"
CONFUSION_MATRIX_FILEPATH_TRAINING = os.path.join(PROGRAM_OUTPUT_DIR, CONFUSION_MATRIX_FILENAME_TRAINING)

CONFUSION_MATRIX_FILENAME_VALIDATION = f"confusion_matrix_lg_validation.pkl"
CONFUSION_MATRIX_FILEPATH_VALIDATION = os.path.join(PROGRAM_OUTPUT_DIR, CONFUSION_MATRIX_FILENAME_VALIDATION)

# CONFUSION_MATRIX_PLOT_FILENAME_TRAINING = f"confusion_matrix_lg_training.png"


if __name__ == "__main__":
    print(os.path.join("resources", "a"))
    print(os.path.exists(DELTA_MATRIX_FILEPATH))
    print(os.path.exists(DELTA_MATRIX_FILEPATH))

