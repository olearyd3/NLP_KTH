import time
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Dmytro Kalpakchi.
"""
class LogisticRegression(object):
    """
    This class performs logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    def __init__(self, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param theta    A ready-made model
        """
        theta_check = theta is not None

        if theta_check:
            self.FEATURES = len(theta)
            self.theta = theta

        #  ------------- Hyperparameters ------------------ #
        self.LEARNING_RATE = 0.1            # The learning rate.
        self.MINIBATCH_SIZE = 256           # Minibatch size
        self.PATIENCE = 5                   # A max number of consequent epochs with monotonously
                                            # increasing validation loss for declaring overfitting
        # ---------------------------------------------------------------------- 


    def init_params(self, x, y):
        """
        Initializes the trainable parameters of the model and dataset-specific variables
        """
        # To limit the effects of randomness
        np.random.seed(524287)

        # Number of features
        self.FEATURES = len(x[0]) + 1

        # Number of classes
        self.CLASSES = len(np.unique(y))

        # Training data is stored in self.x (with a bias term) and self.y
        self.x, self.y, self.xv, self.yv = self.train_validation_split(np.concatenate((np.ones((len(x), 1)), x), axis=1), y)

        # Number of datapoints.
        self.TRAINING_DATAPOINTS = len(self.x)

        # The weights we want to learn in the training phase.
        K = np.sqrt(1 / self.FEATURES)
        self.theta = np.random.uniform(-K, K, (self.FEATURES, self.CLASSES))

        # The current gradient.
        self.gradient = np.zeros((self.FEATURES, self.CLASSES))


        print("NUMBER OF DATAPOINTS: {}".format(self.TRAINING_DATAPOINTS))
        print("NUMBER OF CLASSES: {}".format(self.CLASSES))


    def train_validation_split(self, x, y, ratio=0.9):
        """
        Splits the data into training and validation set, taking the `ratio` * 100 percent of the data for training
        and `1 - ratio` * 100 percent of the data for validation.

        @param x        A (N, D + 1) matrix containing training datapoints
        @param y        An array of length N containing labels for the datapoints
        @param ratio    Specifies how much of the given data should be used for training
        """
        #
        # YOUR CODE HERE
        #
        # set the length of x to N, the total number of samples
        N = len(x)
        # set the training size to be 90% of the samples
        trainingSize = int(N*ratio)
        # random permutation of indices -- create a random set of indices 
        indices = np.random.permutation(N)
        # set each index before the slot corresponding to 'trainingSize' to the trainingIndices list
        trainingIndices = indices[:trainingSize]
        # set each index after the slot corresponding to 'trainingSize' to the validationIndices list
        validationIndices = indices[trainingSize:]
        # set the training inputs to x_train and the outputs to y_train to the output labels for the trainingIndices
        x_train, y_train = x[trainingIndices], y[trainingIndices]
        # do the same with x_val and y_val for the validation set
        x_val, y_val = x[validationIndices], y[validationIndices]
        return x_train, y_train, x_val, y_val


    def loss(self, x, y):
        """
        Calculates the loss for the datapoints present in `x` given the labels `y`.
        """
        #
        # YOUR CODE HERE
        
        # total loss loss is calculated by averaging the loss function across all data points
        # calculate the average of the negatives of the conditional log probabilities
        return np.average(-self.conditional_log_prob(y, x))


    def conditional_log_prob(self, label, datapoint):
        """
        Computes the conditional log-probability log[P(label|datapoint)]
        """
        # label is the output and datapoint is the input

        # most of this is as per the formula from the lecture slides
        # calculate the dot product of theta and x -- BOTH need to be transposed for the dimensions to match
        theta_x = np.dot(np.transpose(self.theta), np.transpose(datapoint))
        # sum all of the theta_x values
        theta_x_sum = np.sum(np.exp(theta_x))
        # softmax equals e to the theta_x over the sum from i = 1 to n of e to the z_i
        # use reshape to put the data into an array (needed for later computations.)
        softmax = np.exp(theta_x) / np.reshape(theta_x_sum, (1, -1))

        # for the first time the function is called, the data is in a list so need to separate it into individually accessible ints
        if (type(label) != int):
            #print("anseo anois oh mo dhia")
            # create an empty probability list
            prob = []
            # for each element in the label
            # method to make it an int
            for i in range(len(label)):
                # append the list 'prob' with the softmax at the slot of 
                # the current value of label slot and each of the values in that label list
                prob.append(softmax[label[i], i])
            # return the log of the probability
            return np.log(prob)
        # if the data are ints (i.e. -- not the first runthrough), return the log probability of the softmax[0][label] slot
        # where label goes from 0-2
        else:
            return np.log(softmax[0][label])


    def compute_gradient(self, minibatch):
        """
        Computes the gradient based on a mini-batch
        """
        #
        # YOUR CODE HERE
        # mostly the same as before
        # most of this is as per the formula from the lecture slides
        # calculate the dot product of theta and x -- BOTH need to be transposed for the dimensions to match
        theta_x = np.matmul(np.transpose(self.theta), np.transpose(self.x[minibatch]))
        # sum all of the theta_x values for each column (indicated by axis=0 -- otherwise sum is wrong)
        theta_x_sum = np.sum(np.exp(theta_x), axis=0)
        # softmax equals e to the theta_x over the sum from i = 1 to n of e to the z_i
        # use reshape to put the data into an array (needed for later computations.)
        softmax = np.exp(theta_x) / np.reshape(theta_x_sum, (1, -1))

        #print(len(minibatch))

        # create an array filled with zeros with self.CLASSES number of rows and length of minibatch (number of iterations) columns
        c = np.zeros(shape=(self.CLASSES, len(minibatch)))
        # fill the array with -1 at the slots where the class is correct -- needed for the gradient of Loss 
        for i in range(len(minibatch)):
            c[self.y[minibatch[i]], i] = -1
            # calculate the gradient of the loss function as the dot product of the datapoints and the softmax + c value divided by the number of values in minibatch
        self.gradient = np.dot(np.transpose(self.x[minibatch]), np.transpose((softmax+c))) / len(minibatch)


    def fit(self, x, y):
        """
        Performs Mini-batch Gradient Descent.
        
        :param      x:      Training dataset (features)
        :param      y:      The list of training labels
        """
        self.init_params(x, y)

        self.init_plot(self.FEATURES)

        start = time.time()
        
        # YOUR CODE HERE
        # initialise the number of iterations and the count to 0
        iteration = 0
        count = 0
        # set the previous validation loss to the loss calculated by the loss function for the validation inputs and outputs
        prev_val_loss = self.loss(self.xv, self.yv)
        # initialise an empty minibatch list
        minibatch = []
        # while the count is less than the value of PATIENCE (for early stopping)
        # i.e. if the validation loss increases for PATIENCE (5) straight times in a row
        # the 'iteration < 20' is to ensure that the loop doesn't exit after an extremely small number due to a random
        # occurrence of the validation loss decreasing without properly fitting the data
        while (count < self.PATIENCE) | (iteration < 20):
            #print("ITERATION:", iteration)
            # same as previously -- # generate datapoints in the range from 0 to the size of training datapoints and append them to the minibatch array
            datapoint = random.randrange(0, self.TRAINING_DATAPOINTS)
            minibatch.append(datapoint)
            #pass minibatch to the compute_gradient function to calculate the gradient
            self.compute_gradient(minibatch)
            # theta is equal to theta minus the learning rate times the gradient
            self.theta = self.theta - self.LEARNING_RATE * self.gradient
            # calculate the validation loss by passing the validation data for x and y to the loss func
            val_loss = self.loss(self.xv, self.yv)
            # if the new validation loss is greater than the previous validation loss add one to count
            if val_loss > prev_val_loss:
                count += 1
                # otherwise reset the count to 0
            else:
                count = 0
            # set the previous validation loss equal to the current validation loss and increase the iteration by 1
            prev_val_loss = val_loss
            iteration = iteration + 1

        print(f"Training finished in {time.time() - start} seconds")


    def get_log_probs(self, x):
        """
        Get the log-probabilities for all labels for the datapoint `x`
        
        :param      x:    a datapoint
        """
        if self.FEATURES - len(x) == 1:
            x = np.array(np.concatenate(([1.], x)))
        else:
            raise ValueError("Wrong number of features provided!")
        return [self.conditional_log_prob(c, x) for c in range(self.CLASSES)]


    def classify_datapoints(self, x, y):
        """
        Classifies datapoints
        """
        confusion = np.zeros((self.CLASSES, self.CLASSES))

        x = np.concatenate((np.ones((len(x), 1)), x), axis=1)

        no_of_dp = len(y)
        for d in range(no_of_dp):
            best_prob, best_class = -float('inf'), None
            for c in range(self.CLASSES):
                prob = self.conditional_log_prob(c, x[d])
                if prob > best_prob:
                    best_prob = prob
                    best_class = c
            confusion[best_class][y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(self.CLASSES)))
        for i in range(self.CLASSES):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(self.CLASSES)))
        acc = sum([confusion[i][i] for i in range(self.CLASSES)]) / no_of_dp
        print("Accuracy: {0:.2f}%".format(acc * 100))


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


def main():
    """
    Tests the code on a toy example.
    """
    def get_label(dp):
        if dp[0] == 1: return 2
        elif dp[2] == 1: return 1
        else: return 0

    from itertools import product
    x = np.array(list(product([0, 1], repeat=6)))

    #  Encoding of the correct classes for the training material
    y = np.array([get_label(dp) for dp in x])

    ind = np.arange(len(y))

    np.random.seed(524287)
    np.random.shuffle(ind)

    b = LogisticRegression()
    b.fit(x[ind][:-20], y[ind][:-20])
    b.classify_datapoints(x[ind][-20:], y[ind][-20:])



if __name__ == '__main__':
    main()
