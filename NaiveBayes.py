__author__ = 'harshad'

import numpy as np
import scipy as sp









def Main():
    matrix = np.loadtxt('data/project3_dataset1.txt')
    
    print type(matrix)
    print len(matrix)
    print len(matrix[0])

if __name__ == '__main__':
    Main()

