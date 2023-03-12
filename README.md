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

### Performance Notes

- using the `pandas` library was not feasible this time since the `read_csv()` function tried to take in the input 
  data all at once and resulted in memory issues (also was quite inefficient in terms of time to read the data)

- for most array operations, using the `auto` chunk size when reading the array from the `.pkl` file leads to the 
  most efficient performance (e.g., `(5792, 5792)` on local machine)

- some array operations in `dask` are more efficient than others...like the `dot()` function compared to the `sum()` 
  function (e.g., to calculate the sum of a given row, it's about `2x` faster to do a dot product of a given column 
  with a vector of ones than just to call `sum()` on the same column of the array)