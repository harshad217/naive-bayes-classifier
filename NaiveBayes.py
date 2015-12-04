__author__ = 'harshad'

import numpy as np
import scipy as sp
import math
from sklearn.naive_bayes import GaussianNB

def calculateGnb(x,mean,var):
    first_term = float(1/math.sqrt(2 * math.pi * var))
    num = math.pow((x - mean),2)
    denom = (2 * var)
    fract = -float(num / denom)
    second_term = math.pow(math.e,fract)
    gnb = first_term * second_term
    return gnb

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

    prior0 = 0
    prior1 = 0
    total_prior = len(training)
    for row in training:
        if row[len(training[0])-1] == 0:
            to_add_row = row[:len(row)-1]
            label0_matrix = np.vstack([label0_matrix,to_add_row])
            prior0 += 1
        elif row[len(training[0])-1] == 1:
            to_add_row = row[:len(row)-1]
            label1_matrix = np.vstack([label1_matrix,to_add_row])
            prior1 += 1

    print 'prior label-0 probability = ', prior0, '/',total_prior
    prior0_probability = float(float(prior0)/float(total_prior))
    prior1_probability = float(float(prior1)/float(total_prior))

    print 'len of matrix of label 0s',len(label0_matrix)
    print 'len of matrix of label 1s',len(label1_matrix)
    print 'label 1s and 0s = ', prior0, prior1

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

    predicted_results = [ ]

    '''predict labels'''
    for i in range(len(testing)):
        label0_gnb_product = prior0_probability
        label1_gnb_product = prior1_probability
        for j in range(len(testing[0])-1):
            #calculate gnb for each feature for label1 and label0 and multiply all
            gnb0 = calculateGnb(testing[i,j],label0_mean_list[0,j],label0_var_list[0,j])
            label0_gnb_product = label0_gnb_product * gnb0
            gnb1 = calculateGnb(testing[i,j],label1_mean_list[0,j],label1_var_list[0,j])
            label1_gnb_product = label1_gnb_product * gnb1

        if gnb0 > gnb1:
            predicted_results.insert(i,0)
        else:
            predicted_results.insert(i,1)

    print len(predicted_results)
    print len(testing_truths)
    accuracy = 0
    for i in range(len(predicted_results)):
        if predicted_results[i] == testing_truths[i]:
            accuracy += 1

    print 'accuracy = ',float(float(accuracy)/float(len(predicted_results)))

    clf = GaussianNB()
    training_to_pass = np.delete(training,axis=1,obj=len(training[0])-1)
    clf.fit(training_to_pass, training_truths)
    testing_to_pass = np.delete(testing,axis=1,obj=len(testing[0])-1)
    pack_results = clf.predict(testing_to_pass)

    acc = 0
    for i in range(len(pack_results)):
        if pack_results[i] == testing_truths[i]:
            acc += 1

    print 'actual accuracy = ', float(float(acc)/float(len(testing_truths)))
if __name__ == '__main__':
    Main()
