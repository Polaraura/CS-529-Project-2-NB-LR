from enum import Enum, auto

import os


class DataFileEnum(Enum):
    INPUT_DATA_TRAINING = auto()
    INPUT_ARRAY_TRAINING = auto()
    INPUT_ARRAY_VALIDATION = auto()
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


class XMatrixType(Enum):
    TRAINING = auto()
    VALIDATION = auto()



# def test(a, b, *args, **kwargs):
def test(a, b, c=1, *args, **kwargs):
    print(a)
    print(b)
    print(c)
    print(args)
    print(kwargs)


def test2(f, *args, **kwargs):
    f(2, 3, *args, **kwargs)


def test3(*args, **kwargs):
# def test(a, b, c=1, *args, **kwargs):
    # print(c)
    print(args)
    print(kwargs)

def test4(f, *args, **kwargs):
    f(*args, **kwargs)


if __name__ == "__main__":
    # a = lambda x: (x, x)
    # b, c = a(1)
    # print(b)
    # print(len(b))

    print(f"-------------------")

    a = (1, 2)
    b = {1: 2}
    test2(test, 1, 2, 4, e=2, g=4)
    test2(test, 1, 2, 4, e=2, g=4)
    test4(test3, 1, 2, a=2, b=4)
    test3(1, 2, a=2, b=4)

    print(f"{(3)}")