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
- `pandas`

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

### Runtime

- each iteration of the gradient descent took ~300ms after the first few initial iterations on local machine (~200ms 
  to update the new weights matrix, ~100ms to normalize weights matrix)
- normalization of weights matrix is needed to avoid values exploding in magnitude when taking the `exp()` when 
  finding the probability matrix (i.e., `overflow error`)

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

### Naive Bayes Implementation

The Naive Bayes portion of the project operates differently than that of Logistic regression due to early design 
differences between the two. 

To run, simply type `python naive_bayes.py -s` with a trailing number. This was implemented using argument parser,
and the majority of arguments are set by default in a way that should be feasible based on the assignment structure. 
Assuming the data download from kaggle is stored as its original name (`cs429529-project-2-topic-categorization`)
in the same directory, and the `resources` and `program_output` folders are pulled locally, no other arguments are needed. 
The command line arguments are named in a way that should be self explanatory, and can be displayed by typing the command
`python naive_bayes.py -h`.

For completeness, the command line arguments are as follows:
- `-trd`,`--train-data`; file path to the `training.csv` file. 
- `-ted`,`--test-data`; file path to the `testing.csv` file. 
- `-trx`,`--train-mtx`; file path to the `training_matrix.npz` sparse matrix file. 
- `-tex`,`--test-mtx`; file path to the `testing_matrix.npz` sparse matrix file. 
- `-vl`,`--vocab-list`; file path to the `vocabulary.txt` file in the downloaded assignment folder. 
- `-nl`,`--news-list`; file path to the `newsgrouplabels.txt` file in the downloaded assignment folder. 
- `-ba`,`--beta-accs`; file path to the `acc.csv` file, which contains beta values and their associated accuracies from kaggle submissions. 
- `-s`,`--scenario`; a number denoting the desired functionality to be performed. options range from 1-5, and are as follows:
  - 1: a single run of Naive Bayes with beta value of 1/vocab size. 
  - 2: generate multiple predictions based on differing beta values to be submitted to kaggle.
  - 3: plot betas vs accuracy from the `acc.csv` file.
  - 4: plot a confusion matrix of the data based on a validation set. 
  - 5: rank words based on their usefulness in making a classification.
- `-ps`,`--print-save`; used to note whether you want output files to be saved or just printed. `False` by default, meaning files will not be saved. 

### Logistic Regression Implementation

Unfortunately, the implementation of Logistic Regression went to numerous overhauls to design and structure so not 
much time was available to add command line args. However, there are a number of quality of life (QOL) features that 
were added to make the training quite seemless. A general overview of the features will be listed out below:

- The main entry of the program is `Main.py` at the root of the repo
- There will be comments and docs for many of the functions in the `logistic_regression/LogisticRegression.py` file, 
  containing the bulk of the implementation so check them out for more details
- A lot of the code relies on `dask` and `sparse` to make the training efficient, so it is critical to have `Python` 
  version 3.9 specifically and not any other version (at least with version `3.10` upwards as that breaks 
  compatability with `dask` and `sparse` for sparse representations)
- All of the main features can be modified in the `Main.py` file, sectioned off in clear groupings
- There are global flags in `Main.py` as well as many other print `DEBUG` flags in the `utilities/Debug.py` file for 
  more information while training is done
- Edit the `Hyperparameters` object for various details, like learning rate (eta), lambda (penalty term), when to 
  print progress updates, when to save intermediate weights (W) matrices, and much more (check 
  `parameters/Hyperparameters` for more details)
- There is an optional progress bar to check how long each iteration or intermediate calculations (when DEBUG flags 
  are set to `True`) -- may increase overall runtime
- A nifty feature is the `MAIN_CUSTOM_TESTING` flag that can go back and list out all major stats of the classifer 
  **without** having to run the entire training process again **if** the corresponding files were saved (can specify 
  which iteration to look at, but may have to edit the hyperparameters manually) -- can reload the W matrix, check 
  the training and validation accuracies again, save predictions at any point where the corresponding matrices are 
  saved, and more (it will also display the total time taken, which is only relevant when training for long periods 
  since getting the stats on a saved W matrix is fairly quick)
- all the related output (not text), like plots and confusion matrices, can be found in the `program_output` folder
- all the saved data (including W, X, delta matrices and more) are saved in the `resources` folder with a detailed 
  file structure (check `logistic_regression/LogisticRegression.py` file for more details)
