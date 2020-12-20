from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import random
import csv

"""
CREATING THE NEURAL NETWORK
"""

def load_dataset():
    """ Loads the dataset from external files and normalize the data. """
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    # The test labels are meaningless,
    # since you're replacing the official MNIST test set with our own test set
    X_test, _ = map(np.array, mndata.load_testing())

    # Remember to center and normalize the data...
    X_train = (X_train-np.mean(X_train,axis=0)) / \
                     (np.std(X_train,axis=0) + 0.00001)
    X_test = (X_test - np.mean(X_test, axis=0)) / \
                     (np.std(X_test, axis=0) + 0.00001)

    # Add in a feature of 1 to every example for the bias
    temp = np.ones((60000, 1))
    X_train = np.concatenate((X_train, temp), axis=1)
    temp = np.ones((10000, 1))
    X_test = np.concatenate((X_test, temp), axis=1)

    return X_train, labels_train, X_test

def splitData(data, labels):
    """ Splits the data (and labels) into training and validation sets. """
    # generate empty matrices to hold training dataset and labels
    X_train = np.zeros((50000, 785))
    labels_train = np.zeros((50000, 1))

    # generate empty matrices to hold validation dataset and labels
    X_val = np.zeros((10000, 785))
    labels_val = np.zeros((10000, 1))

    # indices that we have already seen for training set
    seen = []

    # create the training set
    for i in range(50000):
        temp_index = random.randint(0, 60000)
        while ((temp_index in seen) or (temp_index == 60000)):
            temp_index = random.randint(0, 60000)
        X_train[i] = data[temp_index]
        labels_train[i] = labels[temp_index]
        seen.append(temp_index)

    # create the validation set - we don't need to shuffle validation set
    count = 0
    for i in range(60000):
        if (i not in seen):
            X_val[count] = data[i]
            labels_val[count] = labels[i]
            count = count + 1

    return X_train, labels_train, X_val, labels_val

def trainNeuralNetwork(data, labels, ReLU, ReLU_prime, softmax, \
                       learning_rate=0.001, decay=0.9):
    """ Trains the neural network. """
    # randomly initialize the weights
    V = np.array([np.sqrt(1.0/785) * np.random.randn() for i in range(157000)])
    W = np.array([np.sqrt(1.0/785) * np.random.randn() for i in range(2010)])
    V = np.reshape(V, (200, 785))
    W = np.reshape(W, (10, 201))

    # initialize alpha
    alpha = learning_rate

    # vectorized activation functions
    ReLU_vec = np.vectorize(ReLU)
    ReLU_prime_vec = np.vectorize(ReLU_prime)

    # training loss and classification accuracy for the training set
    training_loss = []
    classification_accuracy = []
    iterations = []

    # stopping criterion used - max iterations
    for i in range(360000):
        # print iteration number
        if (i%1000 == 0):
            print(i)

        # pick a random sample for stochastic gradient descent
        p = data.shape[0]
        while (p == data.shape[0]):
            p = random.randint(0, data.shape[0])
        X_input = data[p]
        X_input = X_input.reshape((1, 785))
        Y = np.zeros((10, 1))
        Y[labels[p][0]] = 1

        # decay
        if ((i is not 0) and (i%120000 == 0)):
            alpha = decay * alpha

        # perform forward pass
        X_hidden, X_output, ReLU_deriv = \
                forward_pass(X_input, V, W, ReLU_vec, softmax, ReLU_prime_vec)

        # perform backward pass
        grad_W, grad_V = back_propagation(X_input, X_hidden, X_output, Y, W, \
                                          ReLU_deriv)

        # update data for the graphs
        if (i%10000 == 0):
            # calculate the current training accuracy and add to set
            cur_labels = predictNeuralNetwork(data, V, W, ReLU_vec, softmax)
            cur_accuracy = computeAccuracy(cur_labels, labels)
            classification_accuracy.append(cur_accuracy)

            # calculate the current training loss and add to set
            cur_loss = computeLoss(Y, X_output)
            training_loss.append(cur_loss)

            # update the iterations
            iterations.append(i)

        # perform stochastic gradient descent update
        W = W - (alpha * grad_W)
        V = V - (alpha * grad_V)

    return V, W, training_loss, classification_accuracy, iterations

def forward_pass(X_input, V, W, activation_hidden, activation_output, h_prime):
    """ Performs a forward pass over the data. """
    S_hidden = np.dot(V, X_input.T)
    X_hidden = activation_hidden(S_hidden)
    X_hidden = X_hidden.reshape((200, 1))

    if (h_prime is not None):
        ReLU_deriv = h_prime(S_hidden)
        ReLU_deriv = ReLU_deriv.reshape((200, 1))
    else:
        ReLU_deriv = None

    # add in bias term
    temp = np.ones((1,1))
    X_hidden = np.concatenate((X_hidden, temp), axis=0)

    S_output = np.dot(W, X_hidden)
    X_output = activation_output(S_output)

    return X_hidden, X_output, ReLU_deriv

def back_propagation(X_input, X_hidden, X_output, Y, W, ReLU_deriv):
    """ Performs a backward pass over the data. """
    delta_output = X_output - Y
    grad_W = np.dot(delta_output, X_hidden.T)
    nobias_W = np.delete(W, W.shape[1]-1, 1)
    delta_hidden = np.dot(nobias_W.T, delta_output)
    delta_hidden = np.multiply(ReLU_deriv, delta_hidden)
    grad_V = np.dot(delta_hidden, X_input)

    return grad_W, grad_V

def ReLU(z):
    """ The activation function for the hidden layer. """
    return max(0, z)

def ReLU_prime(z):
    """ The derivative of the activation function for the hidden layer. """
    if (z > 0):
        return 1
    return 0

def softmax(z):
    """ The activation function for the output layer. """
    e = np.exp(z)
    return e / np.sum(e)

def predictNeuralNetwork(data, V, W, activation_hidden, activation_output):
    """ Generates the predictions for a test set using V and W. """
    results = []
    for i in range(data.shape[0]):
        ignore1, output, ignore2 = forward_pass(data[i], V, W, \
                                   activation_hidden, activation_output, None)
        results.append(output.argmax())
    return results

def computeAccuracy(output, labels):
    """ Computes the accuracy % between predicted and actual labels. """
    # step through the data and count how many differences there are
    num_errors = 0
    for i in range(len(output)):
        if (output[i] != labels[i][0]):
            num_errors = num_errors + 1

    # return the error
    return 100 - (((num_errors * 1.0) / len(output)) * 100)

def computeLoss(labels, g):
    """ Computes the loss based on the provided network. """
    return -1 * np.sum(np.multiply(labels, np.log(g)))

def writeToCSV(labels):
    """ Writes the provided labels to a CSV file for Kaggle submission. """
    with open('results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', \
                 quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Id', 'Category'])
        for i in range(len(labels)):
            first = i+1
            second = int(labels[i])
            writer.writerow([first, second])

"""
RUNNING THE NEURAL NETWORK
"""

# get the data and split it into training and validation sets
X_train, labels_train, X_test = load_dataset()
X_train, labels_train, X_val, labels_val = splitData(X_train, labels_train)

# train the network
V, W, training_loss, classification_accuracy, iterations = \
    trainNeuralNetwork(X_train, labels_train, ReLU, ReLU_prime, \
    softmax, 0.001, 0.9)

ReLU_vec = np.vectorize(ReLU)

# generate the predictions for the training and validation sets
output_train = predictNeuralNetwork(X_train, V, W, ReLU_vec, softmax)
output_val = predictNeuralNetwork(X_val, V, W, ReLU_vec, softmax)

# print the training and validation error
print("The training error is "+str(computeAccuracy(output_train,labels_train))\
      + ".")
print("The validation error is "+str(computeAccuracy(output_val, labels_val))\
      + ".")

# plot the graphs
plt.xlabel("Number of Iterations")
plt.ylabel("Classification Accuracy")
plt.plot(iterations, classification_accuracy)
plt.title("Classification Accuracy vs. Iterations")
plt.show()

plt.xlabel("Number of Iterations")
plt.ylabel("Training Loss")
plt.plot(iterations, training_loss)
plt.title("Training Loss vs. Iterations")
plt.show()

# generate the predictions for the test set
output_test = predictNeuralNetwork(X_test, V, W, ReLU_vec, softmax)

# write the predictions to a CSV file for Kaggle submission
writeToCSV(output_test)
