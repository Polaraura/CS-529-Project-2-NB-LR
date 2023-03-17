# README
## Project 2 - Naive Bayes and Logistic Regression
### CS 529 - Intro to Machine Learning
#### Mike Dinh, Connor Temple, John Tran

### Requirements

The code was run using `Python 3.9` with the following major packages:

- `numpy`
- `scipy`
- `dask`
- `sparse`

### Introduction

The hardest part of this project was loading in the initial input data in a reasonable amount of time and within 
a reasonable amount of memory (so the program wouldn't crash or stall on full RAM).

### Initial Loading of Data

In order to limit the amount of memory used and to read in the input data a **little bit at a time**, the `dask` 
library was used with its "lazy" execution. First, the data was read in as a `dask` `Bag` object, **one line at a 
time**. This resulted in an initial chunk size of `(1, 61190)` corresponding to reading the data one line at a time 
using the `read_text()` function.

The final format we want to manipulate the data as is some form of an **array**, so we mapped each chunk as a 
`numpy` array and then converted that into a `dask` `Array` object. To help save on memory in terms of storage and 
when reading the data, each chunk was converted to a sparse `COO` format using the `sparse` library. At this point, 
the top level object is still a `dask` `Bag` where each chunk is a `dask` `Array` so we stack the individual arrays 
together to form one big `dask` `Array` so we can use normal array operations (e.g., `sum` and `max`).

Finally, to reduce overhead for future execution (i.e., in breaks in between training sessions), the `dask` `Array` 
object was computed and stored into a pickle `.pkl` file as a byte stream that can be read in instead of parsing the 
original input 
data again.

#### Generating Initial Data

In the code, there will be functions with the prefix `generate_` to generate the necessary data (e.g., input data 
and delta matrix) using the `get_data_from_file()` function in `utilities.ParseUtilities` using the `DataFileEnum` 
under `utilities.DataFile`. The saved data files will be stored under the `resources` folder with appropriate names 
(see the `Constants.py` file for more information on the various filenames, filepaths, and constants used).

For the various input data provided to us, we did not store the unzipped data in our `git` repo (data should be 
stored under the 
`cscs429529-project-2-topic-categorization` folder). So, please download all the data and create a folder with the 
same name or use a custom folder name and edit the hardcoded output filepaths in the `Constants.py` file.

### Performance Notes

- using the `pandas` library was not feasible this time since the `read_csv()` function tried to take in the input 
  data all at once and resulted in memory issues (also was quite inefficient in terms of time to read the data)

- for most array operations, using the `auto` chunk size when reading the array from the `.pkl` file leads to the 
  most efficient performance (e.g., `(5792, 5792)` on local machine)

- some array operations in `dask` are more efficient than others...like the `dot()` function compared to the `sum()` 
  function (e.g., to calculate the sum of a given row, it's about `2x` faster to do a dot product of a given column 
  with a vector of ones than just to call `sum()` on the same column of the array)

- `da.dot()` should be used instead of `da.matmul()` because of dealing with chunks of sparse matrices 
  (implementation of `da.matmul()` is **not** compatible with `COO` format used &mdash; gives a "squeeze error"; 
  however, works if the `dask` `Array` is ONE BIG CHUNK and also sparse)

- some calculations conflict if a mix of sparse and dense matrices are used in matrix operations (more specifically, 
  subtraction as both matrices would need to be sparse and have the same fill value) -- had to save the weight 
  matrix and delta matrix as a dense matrix 
