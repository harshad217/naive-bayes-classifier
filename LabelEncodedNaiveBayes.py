__author__ = 'harshad'

import numpy as np
import scipy as sp
import math
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

def calculateGnb(x,mean,var):
    first_term = float(1)/float(math.sqrt(2 * math.pi * var))
    num = math.pow((x - mean),2)
    denom = (2 * var)
    fract = -( float(num) / float(denom) )
    second_term = math.exp(fract)
    gnb = float(first_term) * float(second_term)
    return gnb

def loadData(path):
    file = open(name=path,mode='r')
    content = file.read()
    lines = content.split('\n')
    # print type((lines[0].split())[3])
    cols = []
    for x in range(len(lines)):
        cols.append([])

    matrix = np.ndarray(shape=(len(lines),len((lines[0].split()))))
    col_indexes = set()
    fs = 0
    for i in range(len(lines)):
        features = lines[i].split()
        fs = len(features)
        for j in range(len(features)):
            cols[i].append(features[j])
            if features[j].isalpha():
                col_indexes.add(j)

    print 'cols = ', col_indexes

    for each in col_indexes:
        le = preprocessing.LabelEncoder()
        list_to_pass = []

        for i in range(len(lines)):
            list_to_pass.append(cols[i][each])

        le.fit(list_to_pass)
        print le.classes_
        fitted_list = le.transform(list_to_pass)

        print fitted_list

        for i in range(len(lines)):
            cols[i][each] = fitted_list[i]

    matrix = np.asanyarray(cols,dtype=float)
    return matrix

def Main():
    # matrix = np.loadtxt('data/dataset1.txt')
    # path = 'data/' + raw_input('Enter the file name')
    path = 'data/dataset2.txt'
    matrix = loadData(path)
    original_matrix = matrix
    print matrix[(0,1,2),:]
    # indexes = np.random.permutation(matrix.shape[0])
    indexes = range(matrix.shape[0])
    partition = int(len(matrix) * 0.7)
    training_indexes, testing_indexes = indexes[:partition], indexes[partition:]
    print 'train indexes = ', training_indexes
    training, testing = matrix[training_indexes,:], matrix[testing_indexes,:]
    # print len(training), len(testing)

    training_truths_index = len(training[0]) - 1
    training_truths = training[:,training_truths_index]
    print 'length of train truths = ',len(training_truths)
    training_matrix = training
    training = np.delete(training,training_truths_index,1)
    print 'length of training columns = ', len(training[0])

    testing_truths_index = len(testing[0]) - 1
    testing_truths = testing[:,testing_truths_index]
    print 'length of test truths = ',len(testing_truths)
    testing = np.delete(testing,testing_truths_index,1)
    print 'length of test columns = ', len(testing[0])

    # label0_mean_list = np.zeros(shape=(1,len(training[0])-1))
    label0_mean_list = [ ]
    # label1_mean_list = np.zeros(shape=(1,len(training[0])-1))
    label1_mean_list = [ ]

    label0_matrix = np.empty(shape=(0,len(training[0])-1))
    label1_matrix = np.empty(shape=(0,len(training[0])-1))

    prior0 = 0
    prior1 = 0
    total_prior = len(training)
    print 'shit = ',training_matrix[:,len(training_matrix[0])-1]
    print 'shit complete'
    '''create label0 data matrix and label1 data matrix'''
    label0_matrix = training_matrix[np.ix_(training_matrix[:,len(training_matrix[0])-1] == 0, range(len(training_matrix[0])-1))]

    '''create label0 data matrix and label1 data matrix'''
    label1_matrix = training_matrix[np.ix_(training_matrix[:,len(training_matrix[0])-1] == 1, range(len(training_matrix[0])-1))]

    prior0 = len(label0_matrix)
    prior1 = len(label1_matrix)

    print 'prior label-0 probability = ', prior0, '/',total_prior
    prior0_probability = float(prior0)/float(total_prior)
    prior1_probability = float(prior1)/float(total_prior)

    print 'len of matrix of label 0s',len(label0_matrix)
    print 'len of matrix of label 1s',len(label1_matrix)
    print 'label 1s and 0s = ', prior0, prior1

    '''init label0 mean list'''
    j=0
    for each_column in label0_matrix.T:
        # label0_mean_list[j] = np.mean(each_column)
        label0_mean_list.insert(j,np.mean(each_column))
        j = j + 1

    '''init label1 mean list'''
    j=0
    for each_column in label1_matrix.T:
        # label1_mean_list[j] = np.mean(each_column)
        label1_mean_list.insert(j,np.mean(each_column))
        j = j + 1

    print label0_mean_list, 'len of label0 means = ', len(label0_mean_list)
    print label1_mean_list, 'len of label1 means = ', len(label1_mean_list)

    # label0_var_list = np.zeros(shape=(1,len(training[0])-1))
    label0_var_list = [ ]
    # label1_var_list = np.zeros(shape=(1,len(training[0])-1))
    label1_var_list = [ ]

    '''Init label0 variance list'''
    j=0
    for each_column in label0_matrix.T:
        # label0_var_list[j] = np.var(each_column)
        label0_var_list.insert(j, np.var(each_column))
        j = j + 1

    '''Init label1 variance list'''
    j=0
    for each_column in label1_matrix.T:
        # label1_var_list[j] = np.var(each_column)
        label1_var_list.insert(j, np.var(each_column))
        j = j + 1

    print 'label0 variance list length = ',len(label0_var_list)
    print 'label1 variance list length = ',len(label1_var_list)

    predicted_results = [ ]

    '''predict labels'''
    for i in range(len(testing)):
        label0_gnb_product = prior0_probability
        label1_gnb_product = prior1_probability
        for j in range(len(testing[0])):
            #calculate gnb for each feature for label1 and label0 and multiply all
            gnb0 = calculateGnb(testing[i,j],label0_mean_list[j],label0_var_list[j])
            label0_gnb_product = label0_gnb_product * gnb0
            gnb1 = calculateGnb(testing[i,j],label1_mean_list[j],label1_var_list[j])
            label1_gnb_product = label1_gnb_product * gnb1
            # print 'row:',i,'element:',j,'= ', testing[i,j]

        if label0_gnb_product > label1_gnb_product:
            predicted_results.insert(i,0)
        else:
            predicted_results.insert(i,1)

    print len(predicted_results)
    print len(testing_truths)

    accuracy = 0
    for i in range(len(predicted_results)):
        if predicted_results[i] == testing_truths[i]:
            accuracy += 1

    print 'accuracy = ',float(accuracy)/float(len(predicted_results))

    clf = GaussianNB()
    clf.fit(training, training_truths)
    # testing_to_pass = np.delete(testing,axis=1,obj=len(testing[0])-1)
    pack_results = clf.predict(testing)

    acc = 0
    for i in range(len(pack_results)):
        if pack_results[i] == testing_truths[i]:
            acc += 1

    print 'actual accuracy = ', float(float(acc)/float(len(testing_truths)))

if __name__ == '__main__':
    Main()