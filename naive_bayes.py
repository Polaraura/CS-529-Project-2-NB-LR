import os
import numpy as np
import argparse as ap
from scipy.sparse import csr_matrix, save_npz, load_npz

vocab_size = 61188
class_size = 20
train_size = 12000
test_size = 6774


def parse_arguments():
    parser = ap.ArgumentParser()
    
    parser.add_argument('--train-data', type=ap.FileType(mode='r'), default = './data/training.csv')
    parser.add_argument('--test-data', type=ap.FileType(mode='r'), default = './data/testing.csv')
    parser.add_argument('--train-mtx', default='./data/training_matrix.npz')
    parser.add_argument('--test-mtx', default='./data/testing_matrix.npz') #maybe jsut string type
    parser.add_argument('--vocab-list', type=ap.FileType(mode='r'), default = './data/vocabulary.txt')

    return vars(parser.parse_args())


def parse_matrix(file_path, mtx_path):
    matrix = None

    if os.path.isfile(mtx_path): # If .npz file exists, return its matrix
        mtx = load_npz(mtx_path)
        return mtx.todense()
    
    if 'test' in file_path.name: # initialize columns based on matrix size, 'test' excludes class column.
        matrix = np.zeros((test_size, vocab_size+1), dtype=np.int32)
    else:
        matrix = np.zeros((train_size, vocab_size+2), dtype=np.int32)

    for row, line in enumerate(file_path.readlines()): # fill in the matrix
        matrix[row, :] = list(map(int, line.split(','))) 
    
    matrix = matrix[:, 1:] # remove id column

    mtx = csr_matrix(matrix) # convert to csr matrix to be saved 
    save_npz(mtx_path, mtx) # save csr matrix to file

    return matrix


def words_per_class(matrix):
    word_mtx = np.zeros((class_size, vocab_size+1), dtype=np.int32)

    for i in range(class_size):
        rows = matrix[np.where(matrix[:,-1] == i+1)[0],:]
        word_mtx[i,:-1] = np.sum(rows[:,:-1], axis=0)
        word_mtx[i,-1] = rows.shape[0]
    
    return word_mtx


def probability_matrix(word_matrix, beta=1/vocab_size):
    prob_mtx = np.zeros(word_matrix.shape, dtype=np.float64)
    word_cts = np.sum(word_matrix[:,:-1], axis=1).reshape((word_matrix.shape[0],1))
    document_cts = np.sum(word_matrix[:,-1])

    prob_mtx[:,:-1] = (word_matrix[:,:-1]+beta)/(word_cts+ (beta*vocab_size))
    prob_mtx[:,-1] = word_matrix[:,-1]/document_cts

    return np.log2(prob_mtx)


def classify(test_matrix, prob_matrix, start=12001):
    class_mtx = np.zeros((test_matrix.shape[0],2), dtype=np.int32)
    class_mtx[:,0] = np.arange(test_matrix.shape[0]) + start

    tmp = np.ones((test_matrix.shape[0], test_matrix.shape[1]+1))
    tmp[:,:-1] = test_matrix
    test_matrix = tmp

    product = prob_matrix.dot(test_matrix.T)

    class_mtx[:,1] = np.argmax(product, axis=0) + 1

    return class_mtx


def main():
    input_args = parse_arguments()

    matrix = parse_matrix(input_args.get('train_data'), input_args.get('train_mtx'))
    #test_matrix = parse_matrix(input_args.get('test_data'), input_args.get('test_mtx'))

    word_matrix = words_per_class(matrix)
    prob_matrix = probability_matrix(word_matrix, 1/vocab_size)
    
    matrix = parse_matrix(input_args.get('test_data'), input_args.get('test_mtx'))

    class_matrix = classify(matrix, prob_matrix)
    print(class_matrix)
    print(class_matrix.shape)

main()