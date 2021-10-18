from libsvm.svmutil import svm_load_model, svm_predict, svm_train
import numpy as np
import pandas as pd
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import os
import time
from scipy import spatial
from libsvm.svmutil import *
import sys


def load_and_process(file, class0, class1):
    os.chdir("C:/Users/Shubham/Desktop/Sem7/COL774_ML/Ass_2/mnist")
    f = pd.read_csv(file, sep=',', header=None, index_col=False)
    os.chdir("C:/Users/Shubham/Desktop/Sem7/COL774_ML/Ass_2/Q2")

    data = f[(f.get(784) == class0) | (f.get(784) == class1)]
    # data with class labels 1 and 2 loaded
    print("Data loaded successfully...")
    trainY = data[784]

    # class labels 2 changed to 1, and 1 to -1(+ve and -ve labels)
    trainY = (trainY.apply(lambda x: 1 if (x == class1) else -1))*1.
    trainY.index = np.arange(0, len(data))
    trainX = data.drop(784, axis=1)
    trainX.index = (np.arange(0, len(trainX)))

    trainX /= 255 #scaling greyscale values

    return trainX.to_numpy(), trainY.to_numpy()


def cvxopt_linear(trainX, trainY, c=1):
    negative_index = []
    positive_index = []
    
    m, _ = trainX.shape
    XY = trainY*trainX
    temp = np.dot(XY, XY.T)

    start = time.time()
    G1 = -1*np.eye(m)
    G2 = c*np.eye(m)
    G = cvxopt_matrix(np.vstack((G1, G2)))

    P = cvxopt_matrix(temp)
    q = cvxopt_matrix(-1*np.ones((m, 1)))
    
    h1 = np.zeros(m)
    h2 = np.ones(m)*c
    h = cvxopt_matrix(np.hstack((h1, h2)))

    A = cvxopt_matrix(trainY.reshape(1, -1)*1.)
    b = cvxopt_matrix(np.zeros(1))

    #solver
    solver = cvxopt_solvers.qp(P, q, G, h, A, b)
    alpha = np.array(solver['x'])
    
    S = (alpha > 1e-3)
    support_vec = np.where(S == True)[0]
    S = S.flatten()

    w2 = np.dot((trainY[S] * alpha[S]).T, trainX[S]).reshape(-1, 1)
    bias = trainY[S] - np.dot(trainX[S], w2)
    b2 = np.mean(bias)

    end = time.time()
    print("Time taken to train SVM model(CVXOPT) and Linear Kernel = {:2.3f}sec".format(end-start))

    for i in support_vec:
        if (trainY[i] == -1):
            negative_index.append(i)
        else:
            positive_index.append(i)


    return alpha, w2, b2, support_vec, negative_index, positive_index



def cvxopt_guassian(trainX, trainY, c=1):
    positive_index = []
    negative_index = []
    
    m, _ = trainX.shape
    K = np.zeros((m, m))
    gamma = 0.05
    start = time.time()
    
    pdist = spatial.distance.pdist(trainX, 'sqeuclidean')
    K = np.exp(-1*gamma*spatial.distance.squareform(pdist))
    end = time.time()
    print("Time to Calculate Gaussian Kernel matrix(CVXOPT)= {:2.3f}s".format(end-start))

    # pre-req data
    g1 = -1 * np.eye(m)
    g2 = c*np.eye(m)
    G = cvxopt_matrix(np.vstack((g1, g2)))

    P = cvxopt_matrix(np.outer(trainY, trainY)*K)
    q = cvxopt_matrix(-1*np.ones((m, 1)))

    
    h1 = np.zeros(m)
    h2 = np.ones(m)*c
    h = cvxopt_matrix(np.hstack((h1, h2)))
    A = cvxopt_matrix(trainY.reshape(1, -1)*1.)
    b = cvxopt_matrix(np.zeros(1))

    # solver
    solver = cvxopt_solvers.qp(P, q, G, h, A, b)
    alpha = np.array(solver['x'])

    S = (alpha > 1e-3)

    support_vectors = np.where(S == True)[0]
    S = S.flatten()
    pdist = spatial.distance.pdist(trainX[support_vectors], 'sqeuclidean')
    K_train = np.exp(-1*0.05*spatial.distance.squareform(pdist))
    w_train = np.dot(K_train.T, (alpha[S]*trainY[S]))
    bias = trainY[S] - w_train

    b_train = np.mean(bias)
    print("Time taken to optimize SVM model (CVXOPT) and Gaussian Kernel = {:2.3f}sec".format(time.time()-start))
    for i in support_vectors:
        if (trainY[i] == -1):
            negative_index.append(i)
        else:
            positive_index.append(i)

    return alpha, b_train, support_vectors,  negative_index, positive_index




def predict(trainX, testX, trainY, testY, w, b, kernel, support_vec, alpha):

    # For linear kernel
    if(kernel == 'linear'):
        training_data_predictions = [1.0 if x >= 0 else -1. for x in (np.dot(trainX, w) + b)]
        # accuray calc
        train_accuracy = 0
        for i in range(len(trainY)):
            if (trainY[i] == training_data_predictions[i]):
                train_accuracy += 1
        train_accuracy = (train_accuracy/len(trainY))*100
        
        # for test data
        test_data_predictions = [1. if x >= 0 else -1. for x in (np.dot(testX, w) + b)]
        # accuracy calc
        test_accuracy = 0
        for i in range(len(testY)):
            if (testY[i] == test_data_predictions[i]):
                test_accuracy += 1
        test_accuracy = (test_accuracy/len(testY))*100

    # For Guassian kernel
    elif(kernel == 'gaussian'):
        cdist = spatial.distance.cdist(trainX[support_vec], trainX, 'sqeuclidean')

        K_train = np.exp(-1*0.05*(cdist))
        w = np.dot(K_train.T, (alpha[support_vec] * trainY[support_vec]))

        training_data_predictions = w + b
        training_data_predictions = [1. if x >= 0 else -1. for x in training_data_predictions]

        train_accuracy = 0
        for i in range(len(trainY)):
            if (trainY[i] == training_data_predictions[i]):
                train_accuracy += 1
        train_accuracy = (train_accuracy/len(trainY))*100



        # for test data
        cdist = spatial.distance.cdist(trainX[support_vec], testX, 'sqeuclidean')

        K_test = np.exp(-1*0.05*(cdist))
        w = np.dot(K_test.T, (alpha[support_vec]*trainY[support_vec]))
        test_data_predictions = w + b
        test_data_predictions = [1 if x >= 0 else -
                                 1 for x in test_data_predictions]

        test_accuracy = 0
        for i in range(len(testY)):
            if (testY[i] == test_data_predictions[i]):
                test_accuracy += 1
        test_accuracy = (test_accuracy/len(testY))*100

    return training_data_predictions, test_data_predictions, train_accuracy, test_accuracy


# Entry number = 2018CS10641, class0 = 1, class1 = 2
trainX, trainY = load_and_process(sys.argv[1], 1, 2)
testX, testY = load_and_process(sys.argv[2], 1, 2)

# print(trainX[:2])
# print(trainY[:2])

trainY = trainY.reshape(-1, 1)
m, n = trainX.shape
c = 1  # penalty weight


print("CVXOPT to classify Class 1 and Class 2 with Linear Kernel")
print("Training...")
alpha, w, b, support_vectors, negative_index, positive_index = cvxopt_linear(trainX, trainY, c)


training_data_prediction, test_data_predictions, train_accuracy, test_accuracy = predict(trainX, testX, trainY, testY, w, b, 'linear', support_vectors, alpha)

print("Linear Kernel Results for using CVXOPT...")
print("The number of support vectors are = {}".format(support_vectors.shape[0]))
print("The number of support vectors per class are = {}".format([len(negative_index), len(positive_index)]))
# print("The value of w obtained is =", w)
print("The value of b obtained is = {:2.3f}".format(b))
print("The training accuracy of the model is = {:2.3f}%".format(train_accuracy))
print("The test accuracy of the model is = {:2.3f}%".format(test_accuracy))

print()
print("CVXOPT to classify Class 1 and Class 2 with Guassian Kernel")

alpha, b, support_vectors, negative_index, positive_index = cvxopt_guassian(trainX, trainY, c)

training_data_predictions, test_data_predictions, train_accuracy, test_accuracy = predict(trainX, testX, trainY, testY, w, b, 'gaussian', support_vectors, alpha)


print("Guassian Kernel Results for using CVXOPT...")
print("The number of support vectors are = {}".format(support_vectors.shape[0]))
print("The number of support vectors per class are = {}".format([len(negative_index), len(positive_index)]))
# print("The value of w obtained is =", w)
print("The value of b obtained is = {:2.3f}".format(b))
print("The training accuracy of the model is = {:2.3f}%".format(train_accuracy))
print("The test accuracy of the model is = {:2.3f}%".format(test_accuracy))



# PART C
trainY = trainY.flatten()
testY = testY.flatten()

print("-----------------LIBSVM on Linear Kernel----------------")
start = time.time()
model = svm_train(trainY, trainX, '-t 0 -c 1')
end = time.time()
print("Finished training Linear Kernel SVM in time:", end-start, "sec")

print("Train Accuracy")
train_pred, train_acc, train_val = svm_predict(trainY, trainX, model)
print(train_acc)

print("Test Accuracy")
test_pred, test_acc, test_val = svm_predict(testY, testX, model)
print(test_acc)

print("-----------------LIBSVM on Guassian Kernel----------------")

start = time.time()
svm = svm_train(trainY, trainX, )
model = svm_train(trainY, trainX, '-t 2 -c 1 -g 0.05')
end = time.time()
print("Finished training Guassian Kernel SVM in time:", end-start, "sec")

print("Train Accuracy")
train_pred, train_acc, train_val = svm_predict(trainY, trainX, model)
print(train_acc)

print("Test Accuracy")
test_pred, test_acc, test_val = svm_predict(testY, testX, model)
print(test_acc)
