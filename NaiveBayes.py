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
    print 'length of train truths = ',len(training_truths)
    # training = np.delete(training,training_truths_index,1)
    print 'length of training columns = ', len(training[0])

    testing_truths_index = len(testing[0]) - 1
    testing_truths = testing[:,testing_truths_index]
    print 'length of test truths = ',len(testing_truths)
    # testing = np.delete(testing,testing_truths_index,1)
    print 'length of test columns = ', len(testing[0])

    label0_mean_list = np.zeros(shape=(1,len(training[0])-1))
    label1_mean_list = np.zeros(shape=(1,len(training[0])-1))

    label0_matrix = np.empty(shape=(0,len(training[0])-1))
    label1_matrix = np.empty(shape=(0,len(training[0])-1))

    count0 = 0
    count1 = 0
    for row in training:
        if row[len(training[0])-1] == 0:
            to_add_row = row[:len(row)-1]
            label0_matrix = np.vstack([label0_matrix,to_add_row])
            count0 += 1
        elif row[len(training[0])-1] == 1:
            to_add_row = row[:len(row)-1]
            label1_matrix = np.vstack([label1_matrix,to_add_row])
            count1 += 1

    print 'len of matrix of label 0s',len(label0_matrix)
    print 'len of matrix of label 1s',len(label1_matrix)
    print 'label 1s and 0s = ', count0, count1

    '''init label0 mean list'''
    j=0
    for each_column in label0_matrix.T:
        label0_mean_list[0,j] = np.mean(each_column)
        j = j + 1

    '''init label1 mean list'''
    j=0
    for each_column in label1_matrix.T:
        label1_mean_list[0,j] = np.mean(each_column)
        j = j + 1

    print label0_mean_list, 'len of label0 means = ', len(label0_mean_list[0])
    print label1_mean_list, 'len of label1 means = ', len(label1_mean_list[0])

    label0_var_list = np.zeros(shape=(1,len(training[0])-1))
    label1_var_list = np.zeros(shape=(1,len(training[0])-1))

    '''Init label0 variance list'''
    j=0
    for each_column in label0_matrix.T:
        label0_var_list[0,j] = np.var(each_column)
        j = j + 1

    '''Init label1 variance list'''
    j=0
    for each_column in label1_matrix.T:
        label1_var_list[0,j] = np.var(each_column)
        j = j + 1

    print 'label0 variance list length = ',len(label0_var_list[0])
    print 'label1 variance list length = ',len(label1_var_list[0])
    print label0_var_list


if __name__ == '__main__':
    Main()

