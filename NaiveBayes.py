__author__ = 'harshad'

import numpy as np
import scipy as sp
import math

def calculateGnb(x,mean,var):
    first_term = float(1/math.sqrt(2 * math.pi * var))
    num = math.pow((x - mean),2)
    denom = (2 * var)
    fract = -float(num / denom)
    second_term = math.pow(math.e,fract)
    gnb = first_term * second_term

def Main():
    matrix = np.loadtxt('data/project3_dataset1.txt')
    original_matrix = matrix
    indexes = np.random.permutation(matrix.shape[0])
    partition = int(len(matrix) * 0.7)
    training_indexes, testing_indexes = indexes[:partition], indexes[partition:]
    training, testing = matrix[training_indexes,:], matrix[testing_indexes,:]
    # print len(training), len(testing)

    training_truths_index = len(training[0]) - 1
    training_truths = training[:,training_truths_index]
    print len(training_truths)
    training = np.delete(training,training_truths_index,1)
    print len(training[0])

    testing_truths_index = len(testing[0]) - 1
    testing_truths = testing[:,testing_truths_index]
    print len(testing_truths)
    testing = np.delete(testing,testing_truths_index,1)
    print len(testing[0])


if __name__ == '__main__':
    Main()

