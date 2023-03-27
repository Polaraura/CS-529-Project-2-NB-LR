from enum import Enum, auto

import os

import numpy as np


class DataFileEnum(Enum):
    INPUT_DATA_TRAINING = auto()
    INPUT_DATA_TESTING = auto()
    INPUT_ARRAY_TRAINING = auto()
    INPUT_ARRAY_VALIDATION = auto()
    INPUT_ARRAY_TESTING = auto()
    DELTA_MATRIX = auto()
    X_MATRIX_TRAINING = auto()
    X_MATRIX_VALIDATION = auto()
    X_MATRIX_TESTING = auto()
    W_MATRIX = auto()
    CONFUSION_MATRIX_TRAINING = auto()
    CONFUSION_MATRIX_VALIDATION = auto()


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
    TESTING = auto()


class ClassVectorType(Enum):
    TRAINING = auto()
    VALIDATION = auto()


# def test(a, b, *args, **kwargs):
def test(a, b, c=1, *args, **kwargs):
    print(a)
    print(b)
    print(c)
    print(args)
    print(kwargs)


def another_test(a, b, c=1, **kwargs):
    print(a)
    print(b)
    print(c)
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


def test5(f, **kwargs):
    f(1, 2, **kwargs)


def another_test_2():
    print(f"empty...")


def test6(f, **kwargs):
    f(**kwargs)


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

    print(f"------------")

    test5(another_test, z=3)
    test6(another_test_2)

    print(f"-------------------")

    a = np.array([1, 2, 3])
    b = np.array([1, 4, 3])

    print(np.where(a == b, True, False))

    print(f"{a[0:None]}")

    print(a * b)

    c = np.array([("a", 2), ("b", 2)])

    print(c.dtype)

