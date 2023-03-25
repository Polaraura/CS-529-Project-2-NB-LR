import os
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
import seaborn
import math
import operator as op
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
import pandas as pd

vocab_size = 61188
class_size = 20
train_size = 12000
test_size = 6774


def parse_arguments():
    """
    Parse the command line arguments. most are default based on the data file from kaggle. 
    """
    parser = ap.ArgumentParser()
    
    parser.add_argument('-trd','--train-data', type=ap.FileType(mode='r'), default = './cs429529-project-2-topic-categorization/training.csv')
    parser.add_argument('-ted','--test-data', type=ap.FileType(mode='r'), default = './cs429529-project-2-topic-categorization/testing.csv')
    parser.add_argument('-trx','--train-mtx', default='./resources/training_matrix.npz')
    parser.add_argument('-tex','--test-mtx', default='./resources/testing_matrix.npz') 
    parser.add_argument('-vl','--vocab-list', type=ap.FileType(mode='r'), default = './cs429529-project-2-topic-categorization/vocabulary.txt')
    parser.add_argument('-nl','--news-list', type=ap.FileType(mode='r'), default='./cs429529-project-2-topic-categorization/newsgrouplabels.txt')
    parser.add_argument('-ba','--beta-accs', default='./resources/acc.csv')
    parser.add_argument('-s','--scenario', default='1', help="1: generate a single prediction with beta = 1/vocab. 2: Generate predictions for beta values. 3: Plot betas vs accuracies. 4: Confusion Matrix. 5: Rank Words")
    parser.add_argument('-ps', '--print-save', action='store_true')

    return vars(parser.parse_args())


def parse_matrix(file_path, mtx_path):
    """
    Load the matrix associated with file/mtx path.
    """
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
    """
    Produce a matrix of word counts in each class. Final column contains number of documents in the
    respective class. 
    """
    word_mtx = np.zeros((class_size, vocab_size+1), dtype=np.int32) # initialize 20x61189 matrix

    for i in range(class_size):
        rows = matrix[np.where(matrix[:,-1] == i+1)[0],:] # collects rows from input matrix where class == i+1
        word_mtx[i,:-1] = np.sum(rows[:,:-1], axis=0) # store the sum of word counts in all rows for this class.
        word_mtx[i,-1] = rows.shape[0] # store document count

    return word_mtx


def probability_matrix(word_matrix, beta=1/vocab_size, rank=False):
    """
    Produce a matrix that holds the MAP and MLE estimations.
    """
    prob_mtx = np.zeros(word_matrix.shape, dtype=np.float64)
    word_cts = np.sum(word_matrix[:,:-1], axis=1).reshape((word_matrix.shape[0],1)) # sum of all words in each class
    document_cts = np.sum(word_matrix[:,-1]) # total documents in this sample size, made for use with validation sets.

    if rank: # flag to rank words by importance
        word_rank(word_cts, word_matrix, beta, 100)

    prob_mtx[:,:-1] = (word_matrix[:,:-1]+beta)/(word_cts+ (beta*vocab_size)) # calculate MAP 
    prob_mtx[:,-1] = word_matrix[:,-1]/document_cts # Last column stores MLE

    return np.log2(prob_mtx) 


def word_rank(word_cts, word_mtx, beta, num_words):
    """
    Calculate and output the most important words used when classifying documents. Uses entropy to measure
    importance of words. 
    """
    global words # vocabulary list. initialized earlier to avoid passing around when not needed. 

    word_avg = np.sum(word_cts[:,-1])/vocab_size # take the average words per document
    total_words = np.sum(word_mtx[:,:],axis=0) # total number of each word across all documents. 
    
    std_dev = math.sqrt(np.sum(((total_words[:] - word_avg)**2)[:])/vocab_size) # calculate standard deviation based on square difference

    for i in range(len(total_words)-1): # zero out words if more than 2 standard deviations away
        if total_words[i] < (word_avg - (2*std_dev)) or total_words[i] > (word_avg + (2*std_dev)):
            total_words[i] = 0
        else:
            total_words[i] = 1

    new_word_mtx = np.zeros(word_mtx.shape, dtype=np.float64)
    new_word_mtx = np.multiply(word_mtx, total_words) # zero out words if more than 2 standard deviations away

    entropy = np.zeros(new_word_mtx.shape, dtype=np.float64) # calculate entropy with priors to account for log of zeros. 
    entropy[:,:-1] = ((new_word_mtx[:,:-1] + beta)/ (word_cts + beta)) * np.log2((new_word_mtx[:,:-1] + beta) / (word_cts + beta))
    
    entropy_sums = np.sum(entropy[:-1,:], axis=0) # sum up to get entropy for each word
    sorted_entropy = sorted(entropy_sums.transpose().tolist()) # sort

    min_entropy = sorted_entropy[num_words-1] # establish the cutoff for entropy based on number of rankings

    # List of lists to hold entropy with the associated word.
    ranked_words = [[words[i].rstrip("\n") ,entropy_sums[i] ] for i in range(entropy_sums.shape[0] - 1) if entropy_sums[i] <= min_entropy]

    sorted_words = sorted(ranked_words, key=op.itemgetter(1)) # sort based on entropy

    with open('./program_output/word_rankings.txt','w+') as f: #write results to output file and print
        for i in range(num_words):
            line = "{}. {}, {}\n".format(i+1, sorted_words[i][0], -1*sorted_words[i][1])
            print(line[:-2])
            f.write(line)
    

def classify(test_matrix, prob_matrix, start=12001):
    """
    Produces classifications for the provided test matrix based on the probability matrix. 
    """
    class_mtx = np.zeros((test_matrix.shape[0],2), dtype=np.int32)
    class_mtx[:,0] = np.arange(test_matrix.shape[0]) + start # id column

    tmp = np.ones((test_matrix.shape[0], test_matrix.shape[1]+1))
    tmp[:,:-1] = test_matrix # add a column of 1's to the end of the test_matrix to match dimensions and make MLE count
    test_matrix = tmp

    product = prob_matrix.dot(test_matrix.T) # matrix multiply to get predictions

    class_mtx[:,1] = np.argmax(product, axis=0) + 1 # argmax to get the class with best probability.

    return class_mtx


def plot_confusion_matrix(calc_class, real_class):
    """
    Produce a confusion matrix based on a validation set.
    """
    global news_groups # list of news groups
    conf_matrix = np.zeros((class_size, class_size), dtype=np.int32)

    for row in range(class_size):
        for col in range(class_size):
            conf_matrix[row][col] = np.sum(calc_class[real_class == col+1] == row+1) # check if class is correct

    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Confusion Matrix")

    seaborn.heatmap(conf_matrix, annot=True, fmt=".0f", xticklabels=news_groups, yticklabels=news_groups) # plot
    ax.set_xticklabels(news_groups, rotation=60)

    plt.show()

    print(conf_matrix)

    
def plot_accuracies(betas, accuracies, ps=False):
    """
    Produce a plot of beta values vs accuracy from kaggle data
    """
    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1)
    ax.set_xscale('log')
    ax.set_xlabel("Beta Values")
    ax.set_ylabel("Accuracy")
    ax.set_title("Beta Values vs. Accuracy")
    ax.plot(betas, accuracies, '-bo',)

    if ps:
        plt.savefig("./program_output/beta_accuracy.png",format='png')
    else:
        plt.show()


def basic_nb(train_matrix, test_matrix, ps):
    """
    Perform Naive Bayes to produce a prediction with beta = 1/vocab size.
    """
    word_matrix = words_per_class(train_matrix) # get word counts in each class
    prob_matrix = probability_matrix(word_matrix, 1/vocab_size) # MAP and MLE 

    class_matrix = classify(test_matrix, prob_matrix) # make a classification

    class_matrix1 = [["id", "class"]] + class_matrix.tolist() # add id and class columns for submission
    
    if ps:
        np.savetxt('program_output/predictions.csv', class_matrix1, delimiter=',', fmt = '%s')
    print(class_matrix)


def make_confusion_matrix(matrix):
    """
    Make classifications based on a validation set to produce a confusion matrix.
    """
    train_matrix = matrix[0:9600,:] # split training data for validation
    test_matrix = matrix[9600:,:]
    real_class = test_matrix[:,-1] # get classifications of validation set

    word_matrix = words_per_class(train_matrix) # get word counts in each class

    beta_estimates = {}
    accuracies = {}

    beta = 1/vocab_size
    prob_matrix = probability_matrix(word_matrix, beta) # MAP and MLE 
    class_matrix = classify(test_matrix[:,:-1], prob_matrix, (train_matrix.shape[0]+1)) # make classification 
    calc_class = class_matrix[:,1].reshape((real_class.size,1)) # gather the classes without indices

    plot_confusion_matrix(calc_class, real_class) # plot the confusion matrix

    # betas = np.geomspace(0.00001, 1.0, 6) # using 6 beta values, increasing by factor of 10 each time.
    # for i in betas: # perform typical operations for each beta value. 
    #     prob_matrix = probability_matrix(word_matrix, i) # MAP and MLE for Beta values
    #     beta_estimates[i] = classify(test_matrix[:,:-1], prob_matrix, (train_matrix.shape[0] + 1)) # store class with associated beta
    #     calc_class = beta_estimates[i][:,1].reshape((real_class.size, 1)) # get the classification
    #     accuracies[i] = np.sum(real_class == calc_class)/real_class.size # calculate associated accuracies
    
    # best_beta = max(accuracies, key=accuracies.get) # get the best beta value to use for the confusion matrix.
    # calc_class = beta_estimates[best_beta][:,1].reshape((real_class.size,1)) 

    # plot_confusion_matrix(calc_class, real_class) # plot the confusion matrix


def optimized_nb(train_matrix, test_matrix, ps):
    """
    Produces predictions of the training set for various beta values.
    """
    word_matrix = words_per_class(train_matrix) # get word counts for each class.

    betas = np.geomspace(0.00001, 1.0, 6) # set up betas
    for i in betas: # run for each beta
        prob_matrix = probability_matrix(word_matrix, i) # MAP and MLE
        class_matrix = classify(test_matrix, prob_matrix) 
        class_matrix1 = [["id", "class"]] + class_matrix.tolist()
        if ps:
            np.savetxt("program_output/beta_{}_prediction.csv".format(i), class_matrix1, delimiter=',', fmt='%s')
        print(class_matrix)


def optimized_plot(file_path, ps):
    """
    Plot betas vs accuracy from file containing kaggle scores with associated beta values. 
    """
    data = pd.read_csv(file_path) # read in the data, put each in a list
    betas = data['beta'].tolist() 
    accuracies = data['accuracy'].tolist()
    
    plot_accuracies(betas, accuracies, ps) # plot 


def rank_words(train_matrix, test_matrix, ps):
    """
    Produces a ranking of words based on their usefulness in making a classification. 
    """
    word_matrix = words_per_class(train_matrix) # get word counts for each class
    prob_matrix = probability_matrix(word_matrix, 1/vocab_size, True) # MAP and MLE, True causes ranking to happen

    class_matrix = classify(test_matrix, prob_matrix) # not actually used

    class_matrix1 = [["id", "class"]] + class_matrix.tolist()


def main():
    """
    Main method, runs appropriate function based on input parameters. 
    """
    input_args = parse_arguments()
    train_matrix = parse_matrix(input_args.get('train_data'), input_args.get('train_mtx')) # get training matrix, since its used everywhere

    if input_args.get('scenario') == '1':
        print("doing a default run with beta=1/vocab to generate a solution file")
        # load the test matrix for use
        test_matrix = parse_matrix(input_args.get('test_data'), input_args.get('test_mtx'))
        basic_nb(train_matrix, test_matrix, input_args.get('print_save'))

    elif input_args.get('scenario') == '2':
        print("generating out files for various beta values for obtaining accuracy scores.")
        # load test matrix for use
        test_matrix = parse_matrix(input_args.get('test_data'), input_args.get('test_mtx'))
        optimized_nb(train_matrix, test_matrix, input_args.get('print_save'))

    elif input_args.get('scenario') == '3':
        print("Using previously gathered accuracies from the test set with various beta values to plot the results.")

        optimized_plot(input_args.get('beta_accs'), input_args.get('print_save'))

    elif input_args.get('scenario') == '4':
        print("Using a validation set to create the confusion matrix.")

        global news_groups # make global to avoid passing it as arguments everywhere
        news_groups = input_args.get('news_list').readlines()

        make_confusion_matrix(train_matrix)

    elif input_args.get('scenario') == '5':
        print("Ranking words")

        global words # make global to avoid passing it as arguments everywhere
        words = input_args.get('vocab_list').readlines()

        test_matrix = parse_matrix(input_args.get('test_data'), input_args.get('test_mtx'))
        rank_words(train_matrix, test_matrix, input_args.get('print_save'))


if __name__ == "__main__":
    main()