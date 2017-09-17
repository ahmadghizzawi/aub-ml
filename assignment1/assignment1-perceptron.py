import numpy as np
import random
from numpy import sign as sign
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, n, n2=10, runs=1000):
        self.N = n

        # create random line as target function f(x)
        point1 = [1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        point2 = [1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)]

        x1A = point1[1]
        x2A = point1[2]

        x1B = point2[1]
        x2B = point2[2]

        # create the target function weights based on the random points
        self.target_weights = np.array([x1B * x2A - x1A * x2B, x2B - x1A, x1A - x1B])

        # initialize training set and their correct labels
        self.training_set, self.labels = self.generate_training_set()

        # initialize out_of_sample set
        self.test_set = np.random.uniform(-1, 1, size=(10000, 2))
        x0 = np.ones((10000, 1))
        self.test_set = np.insert(self.test_set, [0], x0, axis=1)

        # initialize best_perceptron
        self.best_perceptron = None

        # run the model with subset of the training_set
        self.run(n2, runs)

        # run the model with the whole dataset
        self.run(n, runs)

    def generate_training_set(self):
        """Generates an n by 3 uniformally distributed dataset"""

        # Generate random uniformally distributes points
        training_set = np.random.uniform(-1, 1, size=(self.N, 2))

        # Insert x0 values into the begining of the array.
        x0 = np.ones((self.N, 1))
        training_set = np.insert(training_set, [0], x0, axis=1)

        # Generate labels for the training set
        labels = []
        [labels.append(Perceptron.apply(self.target_weights, point)) for point in training_set]
        labels = np.array([labels]).T

        # Set training_set and labels as instance attributes
        return training_set, labels

    @staticmethod
    def apply(w, x):
        # apply h(x)
        return sign(np.dot(w, x))

    @staticmethod
    def learn(w, x, label):
        # learn from misclassifications
        return w + label * x

    def plot(self, set_size):
        # plot the set based on the set_size provided
        labels = self.labels[0:set_size]
        training_set = self.training_set[0:set_size]

        # change the axis to fit our dataset
        plt.axis([-1, 1, -1, 1])

        # breakdown the dataset into two separate arrays, each array representing their label by f(x)
        training_set_above_line = []
        training_set_below_line = []

        for i in range(len(labels)):
            if labels[i] == 1:
                training_set_above_line.append(training_set[i])
            else:
                training_set_below_line.append(training_set[i])

        training_set_above_line = np.array(training_set_above_line)
        training_set_below_line = np.array(training_set_below_line)

        # plot the sets
        if training_set_above_line.size > 0:
            training_set_x1_1 = training_set_above_line[:, 1]
            training_set_x2_1 = training_set_above_line[:, 2]
            plt.scatter(training_set_x1_1, training_set_x2_1, c='b')

        if training_set_below_line.size > 0:
            training_set_x1_neg_1 = training_set_below_line[:, 1]
            training_set_x2_neg_1 = training_set_below_line[:, 2]
            plt.scatter(training_set_x1_neg_1, training_set_x2_neg_1, c='r')

        # generate 50 evenly spaced numbers from -1 to 1.
        line = np.linspace(-1, 1)

        # plot the f(x) in blue
        m, b = -self.target_weights[1] / self.target_weights[2], -self.target_weights[0] / self.target_weights[2]
        plt.plot(line, m * line + b, 'b-', label='f(x)')

        # plot the g(x) in dashed red
        m1, b1 = -self.best_perceptron[1] / self.best_perceptron[2], -self.best_perceptron[0] / self.best_perceptron[2]
        plt.plot(line, m1 * line + b1, 'r--', label='h(x)')

        plt.show()

    def pla(self, set_size):
        '''Perceptron Learning Alogrithm '''
        # Take a subset of the data set based on the passed set_size
        training_set = self.training_set[0:set_size]
        correct_labels = self.labels[0:set_size]

        def apply_h(w):
            # labels dataset using the hypothesis's weight
            pla_labels = []
            for i in range(len(training_set)):
                pla_labels.append(Perceptron.apply(w, training_set[i]))
            return pla_labels

        def misclassified(pla_labels):
            # returns a list of indexes of misclassfied points based on the last hypothesis found.
            misclassified = []
            for i in range(set_size):
                if self.labels[i] != pla_labels[i]:
                    misclassified.append(i)
            return misclassified

        # initate weights vector to 0s
        w = [0, 0, 0]

        # apply the algo to the dataset with weights = 0.
        # All points are initially misclassified.
        pla_labels = apply_h(w)

        # generate the set of misclassified points
        misclassfied_indexes = misclassified(pla_labels)

        total_iterations = 1
        total_incorrect = len(misclassfied_indexes)

        while len(misclassfied_indexes) > 0:
            # pick a random index to learn from.
            rand_index = random.choice(misclassfied_indexes)
            w = Perceptron.learn(w, training_set[rand_index], correct_labels[rand_index])

            # apply the algo on updated weights
            pla_labels = apply_h(w)

            # generate a new list of misclassified indexes.
            misclassfied_indexes = misclassified(pla_labels)

            total_iterations += 1
            total_incorrect += len(misclassfied_indexes)

        # best_perceptron has the least difference
        difference = self.evaluate_difference(w)
        if difference < self.best_perceptron[3]:
            self.best_perceptron = np.hstack((w, difference))

        return total_iterations, total_incorrect

    def evaluate_difference(self, hypothesis):
        # evaluate f and g on a sample of 10000 out of sample points.
        sample = self.test_set
        sample_size = len(sample)

        i = 0
        total_misclassified = 0
        while i < sample_size:
            x = sample[i]
            target_classification = Perceptron.apply(self.target_weights, x)
            hypothesis_classification = Perceptron.apply(hypothesis, x)
            if target_classification != hypothesis_classification:
                total_misclassified += 1
            i += 1

        return total_misclassified / sample_size

    @staticmethod
    def display_figures(set_size, average_iterations, difference):
        print('Average iterations to converge for dataset N=' + str(set_size) + ': ' + str(average_iterations))
        print('Average P[f(x) != g(x)]: ' + str(difference))

    def run(self, size, runs):
        print('Running for dataset N=' + str(size)
              + ' ' +
              str(runs)
              + ' times and evaluating each hypothesis againt a 10,000 test_set.. This should take around a minute.')

        # reinitialize best_perceptron
        self.best_perceptron = [0, 0, 0, 100000]

        total_iterations = 0
        i = 0
        while i < runs:
            result = self.pla(size)
            total_iterations += result[0]
            i += 1
        average_iterations = total_iterations / runs

        # evaluate difference between f and g on an out of sample data
        difference = self.evaluate_difference(self.best_perceptron[0:3])

        self.plot(size)
        Perceptron.display_figures(size, average_iterations, difference)


perceptron = Perceptron(100, n2=10, runs=1000)
