from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from utilities.ParseUtilities import \
    parse_class_labels, \
    parse_data_training_array, \
    save_da_array_pickle, \
    load_da_array_pickle

from utilities.Constants import \
    CLASS_LABELS_FILENAME, \
    CLASS_LABELS_FILEPATH

# Global variables
MAIN_DEBUG = False
MAIN_PRINT = True


if __name__ == "__main__":
    class_labels_dict = parse_class_labels(CLASS_LABELS_FILEPATH)

    print(f"class labels dict: {class_labels_dict}")

