from enum import Enum, auto

import os


class DataFileEnum(Enum):
    OUTPUT_ARRAY_TRAINING = auto()
    DELTA_MATRIX = auto()
    X_MATRIX = auto()
    W_MATRIX = auto()


class DataOptionEnum(Enum):
    SAVE = auto()
    LOAD = auto()


class WMatrixOptionEnum(Enum):
    NO_NORMALIZATION = auto()
    W_NORMALIZATION = auto()
    X_NORMALIZATION = auto()
    W_X_NORMALIZATION = auto()


class XMatrixOptionEnum(Enum):
    NO_NORMALIZATION = auto()
    X_NORMALIZATION = auto()

