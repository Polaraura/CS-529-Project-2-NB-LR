from utilities.DataFile import WMatrixOptionEnum
import os

CHUNK_SIZE = 100

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_SEP = os.path.sep
DIR_UP = f".."
OUTPUT_DIR = f"resources"

INPUT_FILEPATH_TRAINING = f"cs429529-project-2-topic-categorization{PATH_SEP}training.csv"

OUTPUT_FILENAME_TRAINING = f"output_array_training.pkl"
# OUTPUT_FILEPATH_TRAINING = f"{OUTPUT_DIR}{OUTPUT_FILENAME_TRAINING}"
OUTPUT_FILEPATH_TRAINING = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_TRAINING)
OUTPUT_CHUNK = (1, 61190)
OUTPUT_SHAPE = (12000, 61190)

CLASS_LABELS_FILENAME = f"newsgrouplabels.txt"
CLASS_LABELS_FILEPATH = f"cs429529-project-2-topic-categorization{PATH_SEP}{CLASS_LABELS_FILENAME}"

DELTA_MATRIX_FILENAME = f"delta_matrix.pkl"
DELTA_MATRIX_FILEPATH = f"{OUTPUT_DIR}{PATH_SEP}{DELTA_MATRIX_FILENAME}"

X_MATRIX_FILENAME = f"x_matrix.pkl"
X_MATRIX_FILEPATH = f"{OUTPUT_DIR}{PATH_SEP}{X_MATRIX_FILENAME}"

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

