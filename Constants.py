from utilities.DataFile import WMatrixOptionEnum, XMatrixOptionEnum
from utilities.FileSystemUtilities import create_sub_directories

import os

CHUNK_SIZE = 100

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_SEP = os.path.sep
DIR_UP = f".."

# TODO: added the very top level for specifying training/validation split
# FIXME: too many dependencies to change...hard code "validation_split=0.2"
OUTPUT_DIR = os.path.join(f"resources", f"validation_split=0.2")
create_sub_directories(OUTPUT_DIR)

INPUT_DATA_FILEPATH_TRAINING = f"cs429529-project-2-topic-categorization{PATH_SEP}training.csv"

INPUT_ARRAY_FILENAME_ENTIRE_DATA = f"entire_input_data.pkl"
INPUT_ARRAY_FILEPATH_ENTIRE_DATA = os.path.join(OUTPUT_DIR, INPUT_ARRAY_FILENAME_ENTIRE_DATA)

INPUT_ARRAY_FILENAME_TRAINING = f"training_array.pkl"
# OUTPUT_FILEPATH_TRAINING = f"{OUTPUT_DIR}{OUTPUT_FILENAME_TRAINING}"
INPUT_ARRAY_FILEPATH_TRAINING = os.path.join(OUTPUT_DIR, INPUT_ARRAY_FILENAME_TRAINING)

INPUT_ARRAY_FILENAME_VALIDATION = f"validation_array.pkl"
INPUT_ARRAY_FILEPATH_VALIDATION = os.path.join(OUTPUT_DIR, INPUT_ARRAY_FILENAME_VALIDATION)

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

W_MATRIX_FILENAME_WITHOUT_EXTENSION = f"w_matrix"
W_MATRIX_FILENAME_EXTENSION = f".pkl"
# W_MATRIX_FILEPATH = f"{OUTPUT_DIR}{W_MATRIX_FILENAME}"
W_MATRIX_TOP_DIR = f"w_matrix"
W_MATRIX_SUB_DIR_DICT = {WMatrixOptionEnum.NO_NORMALIZATION: f"no_normalization",
                         WMatrixOptionEnum.W_NORMALIZATION: "w_normalization",
                         WMatrixOptionEnum.X_NORMALIZATION: f"x_normalization",
                         WMatrixOptionEnum.W_X_NORMALIZATION: f"w_x_normalization"}


if __name__ == "__main__":
    print(os.path.join("resources", "a"))
    print(os.path.exists(DELTA_MATRIX_FILEPATH))
    print(os.path.exists(DELTA_MATRIX_FILEPATH))

