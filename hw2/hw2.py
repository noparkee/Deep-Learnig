import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg

# you can define/use whatever functions to implememt

########################################
# Part 1. cross entropy loss
########################################
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return cross entropy loss
    # result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x=np.reshape(x.T, (-1, n))

    # feat_dim  # 2     
    # x.shape   # 2x400 - input, 2차원 feature 데이터 400개
    # y.shape   # 400,  - output, 400 개의 데이터 
    # W         # 4x2   - class 갯수 x feature dimesion
    # b.shape   # 4x1   - class 갯수만큼 bias
    # n         # 400   - data train 데이터
   
    s = W @ x + b       # 4x400 class 4개와 데이터 400개 - linear score
    s = np.exp(s)       # - exponential
    
    i_sum = 0
    for i in range (n):     # - normalize
        for j in range (num_class):
            i_sum += s[:,i][j]
        for j in range (num_class):
            s[:,i][j] = s[:,i][j] / i_sum

    one_hot = np.eye(num_class)[y]
    
    sum = 0
    for i in range (n):
        sum += np.dot(one_hot[i], np.log(s[:,i]))       # - log 씌워서 내적
    
    return -sum / n
        
    

########################################
# Part 2. SVM loss calculation
########################################
def svm_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return SVM loss
    # SVM loss를 return하는데, 데이터셋의 평균 loss!
    # result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1, n))
    
    # feat_dim  # 2     
    # x.shape   # 2x400 - input, 2차원 feature 데이터 400개
    # y.shape   # 400,  - output, 400 개의 데이터 
    # W         # 4x2   - class 갯수 x feature dimesion
    # b.shape   # 4x1   - class 갯수만큼 bias
    # n         # 400   - data train 데이터
   
    s = W @ x + b       # 4x400 class 4개와 데이터 400개
    
    sum = 0
    for i in range (n):
        temp = s[:,i]
        for j in range (num_class):
            yi = y[i]
            if yi != j:
                if 0 < temp[j] - temp[yi] + 1:
                    sum += temp[j] - temp[yi] + 1

    avg_loss = sum / n
    
    return avg_loss


########################################
# Part 3. kNN classification
########################################
def knn_test(X_train, y_train, X_test, y_test, n_train_sample, n_test_sample, k):
    # implement your function here
    #return accuracy
    
    # 100개의 test 데이터에 대해서 400개의 train 데이터와 거리 
    dists = -2 * np.dot(X_test, X_train.T) + np.sum(X_train**2, axis = 1) + np.sum(X_test**2, axis = 1)[:, np.newaxis]

    correct = 0
    freq = []

    for i in range (n_test_sample):         # i번째 test 데이터와~
        # error 작은 순으로 index, s[0]은 test 데이터와 가장 가까운 train 데이터의 인덱스 값
        s = dists[i].argsort()
        # error 작은 순으로 dists 
        dists[i][s]   

        for j in range (k):
            freq.append(y_train[s[j]])      # 상위 k개 택
        
        m = stats.mode(freq)        # freq 중 최대 빈도
        
        if y_test[i] == m[0]:
            correct = correct + 1       # correct 갯수 더하기
        
        freq.clear()        # freq 비우기

    accuracy = correct / n_test_sample

    return accuracy


# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x_te, y_te, num_class,n_test):
    Wb = np.reshape(Wb, (-1, 1))
    dlen = len(x_te[0])
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * dlen)], (num_class, dlen))
    accuracy = 0;

    for i in range(n_test):
        # find the linear scores
        s = W @ x_te[i].reshape((-1, 1)) + b
        # find the maximum score index
        res = np.argmax(s)
        accuracy = accuracy + (res == y_te[i]).astype('uint8')

    return accuracy / n_test

# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.0

print('number of classes: ',num_class,' sigma for data scatter:',sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# set classifiers to 'svm' to test SVM classifier
# set classifiers to 'softmax' to test softmax classifier
# set classifiers to 'knn' to test kNN classifier
classifiers = 'svm'

if classifiers == 'svm':
    print('training SVM classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    print('testing SVM classifier...')

    Wb = result.x
    print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

elif classifiers == 'softmax':
    print('training softmax classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')

    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

else:  # knn
    # k value for kNN classifier. k can be either 1 or 3.
    k = 3
    print('testing kNN classifier...')
    print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, n_train, n_test, k)*100
          , '% for k value of ', k)
