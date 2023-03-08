from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import dask.bag as db
import dask.array as da


class LogisticRegression:
    def __init__(self,
                 input_array: da.array):
        self.input_array = input_array[:, 1:-1]

        # number of rows is the number of examples
        # number of columns is the total number of attributes (except for the index column at the beginning and the
        # class column at the end)
        self.m, self.n = self.input_array




    def set_initial_parameters(self):
        pass
