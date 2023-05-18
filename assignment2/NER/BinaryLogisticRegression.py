from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from alive_progress import alive_bar

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.01  # The learning rate.
    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
    MAX_ITERATIONS = 100 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000 # Minibatch size (only for minibatch gradient descent)

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)



    # ----------------------------------------------------------------------


    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + math.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """

        # REPLACE THE COMMAND BELOW WITH YOUR CODE
        # conditional probability for positive case is the sigmoid using θ Transpose x
        conditionalProb = self.sigmoid(np.dot(np.transpose(self.theta), self.x[datapoint]))
        # for negative case return 1- conditional prob
        if label == 0:
            return 1 - conditionalProb
            # positive case
        else:
            return conditionalProb

    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """

        # YOUR CODE HERE
        # calculate the sigmoid part of the summation since h_theta(x) = sigmoid(theta T x)
        h = 1.0 / (1 + np.exp(-np.dot(self.theta, np.transpose(self.x))))
        # set the gradient equal to (1/number of data points) times the dot of h-y and the datapoint
        self.gradient = (1/self.DATAPOINTS)*np.dot((h-self.y), self.x)     

    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """
        
        # YOUR CODE HERE
        # same as compute_gradient
        h = 1.0 / (1 + np.exp(-np.dot(self.theta, np.transpose(self.x[minibatch]))))
        self.gradient = np.dot((h-self.y[minibatch]), self.x[minibatch])/self.MINIBATCH_SIZE
        

    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """

        # YOUR CODE HERE
        # same as compute_gradient_for_all but for a single datapoint
        h = 1.0 / (1 + np.exp(-np.dot(self.theta, np.transpose(self.x[datapoint]))))
        self.gradient = (h - self.y[datapoint])*self.x[datapoint]

    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)
        # initialise a vector for iterating to 0
        datapoint = random.randint(0, self.DATAPOINTS-1)
        self.compute_gradient(datapoint)
        iteration = 0
        # YOUR CODE HERE
        # repeat the loop for MAX_ITERATIONS times or until convergence has been reached
        while any(np.abs(self.gradient) > self.CONVERGENCE_MARGIN) and iteration < self.MAX_ITERATIONS*self.DATAPOINTS:
            # randomly select an integer from 0 to the number of datapoints
            datapoint = random.randint(0, self.DATAPOINTS-1)
            # call the helper function for computing the gradient for stochastic gradient descent
            self.compute_gradient(datapoint)
            # calculating theta, as per the algorithm theta = theta - alpha*gradient
            self.theta = self.theta - self.LEARNING_RATE * self.gradient
            # print the graph
            #self.update_plot(np.sum(np.square(self.gradient)))
            print("Gradient:", self.gradient)
            # update the iteration count
            iteration = iteration + 1

    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        # YOUR CODE HERE

        minibatch = []
        # generate datapoints in the range from 0 to the size of datapoints and append them to the minibatch array
        # do this for the size of the minibatch (1000 here)
        for i in range(self.MINIBATCH_SIZE):
            datapoint = random.randrange(0, self.DATAPOINTS)
            minibatch.append(datapoint)
        # use the helper function to compute the gradient for the minibatch
        self.compute_gradient_minibatch(minibatch)

        # initialise a variable to keep track of the number of iterations completed
        iteration = 0

        # while the convergence has not been reached
        while any(np.abs(self.gradient) > self.CONVERGENCE_MARGIN):
            #  generate 1000 random datapoints to use for the minibatch
            for i in range(self.MINIBATCH_SIZE):
                datapoint = random.randrange(0, self.DATAPOINTS)
                minibatch[i] = datapoint
            # call the helper function to calculate the summation part
            self.compute_gradient_minibatch(minibatch)
            # calculate theta same as before
            self.theta = self.theta - self.LEARNING_RATE * self.gradient
            # plot the graph and print the gradients
            if iteration % 10 == 0:
                self.update_plot(np.sum(np.square(self.gradient)))
                print("Gradient:", self.gradient)
            iteration = iteration + 1

    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        # set a small value for the learning rate hyperparameter alpha
        self.init_plot(self.FEATURES)
        # YOUR CODE HERE
        # call the helper function and set the iteration count to 0
        self.compute_gradient_for_all()
        i = 0
        # any -- because there is more than one element in the array, so if any of values in gradient are greater than 0.001, 
        # convergence hasn't happened yet
        while any(np.abs(self.gradient) > self.CONVERGENCE_MARGIN):
            # call helper function to compute the gradient = sum.... line from the algorithm
            self.compute_gradient_for_all()
            # theta[k] = theta[k] - alpha*gradient[k] -- from algorithm
            self.theta = self.theta - self.LEARNING_RATE*self.gradient
            # update the graph and print the gradient value on every 10th iteration
            if i % 10 == 0:
                self.update_plot(np.sum(np.square(self.gradient)))
                print("Gradient:", self.gradient)
            i += 1

    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:');

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))
        # added in these lines
        # accuracy = (true positive + true negative)/All
        print("\nAccuracy:", (confusion[0][0] + confusion[1][1]) / np.sum(confusion))
        # precision is the percentage of items that the system detected positive that are actually positive
        # precision = (true positive)/(true positive + false positive)
        if(confusion[0][0] + confusion[0][1] != 0):
            print("Precision for No Name class:", confusion[0][0] / (confusion[0][0] + confusion[0][1]))
        else:
            print("Precision for No Name class: WARNING! No classifications found for Name")
        if(confusion[1][1] + confusion[1][0] != 0):
            print("Precision for Name class:", confusion[1][1] / (confusion[1][1] + confusion[1][0]))
        else: 
            print("Precision for Name class: WARNING! No classifications found for No Name")
        # recall is the percentage of items actually present in the system that were correctly identified
        # recall = (true positive)/(true positive + false negative)
        if(confusion[0][0] + confusion[1][0] != 0):
            print("Recall for ""No Name"" class:", confusion[0][0] / (confusion[0][0] + confusion[1][0]))
        else:
            print("Precision for No Name class: WARNING! No correct classifications found for No Name")
        if(confusion[1][1] + confusion[0][1] != 0):
            print("Recall for Name class:", confusion[1][1] / (confusion[1][1] + confusion[0][1]))
        else: 
            print("Precision for Name class: WARNING! No correct classifications found for Name")

    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]
    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()
