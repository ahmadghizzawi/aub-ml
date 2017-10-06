import numpy as np
from numpy import sign as sign
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


class Utils:
    @staticmethod
    def generate_noise(labels, percentage_noise):
        # percentage of the dataset to be noisy.
        total_noisy = len(labels) * percentage_noise
        i = 0
        while i < total_noisy:
            # choose random index from the dataset
            rand_index = random.randint(0, len(labels) - 1)
            # reverse the label
            labels[rand_index] *= -1
            i += 1

        return labels

    @staticmethod
    def generate_set(N):
        """Generates an n by 3 uniformly distributed dataset"""

        # Generate random uniformally distributes points
        training_set = np.random.uniform(-1, 1, size=(N, 2))

        # Insert x0 values into the begining of the array.
        x0 = np.ones((N, 1))
        training_set = np.insert(training_set, [0], x0, axis=1)

        # Set training_set and labels as instance attributes
        return training_set

    @staticmethod
    def display_figures(num_of_runs, ein_total, eout_total):
        print('Average Ein(g): ' + str(ein_total / num_of_runs))
        print('Average Eout(g): ' + str(eout_total / num_of_runs))

    @staticmethod
    def plot(training_set, labels, target_weights, w, transformation=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        # change the axis to fit our dataset
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

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
            ax.scatter(training_set_x1_1, training_set_x2_1, c='b')

        if training_set_below_line.size > 0:
            training_set_x1_neg_1 = training_set_below_line[:, 1]
            training_set_x2_neg_1 = training_set_below_line[:, 2]
            ax.scatter(training_set_x1_neg_1, training_set_x2_neg_1, c='r')

        # generate 50 evenly spaced numbers from -1 to 1.
        x = np.linspace(-1, 1)

        # plot the f(x) in blue
        ax.add_patch(
            patches.Circle(
                (0, 0),  # (x,y)
                0.775,  # radius
                fill=False
            )
        )
        if not transformation:
            # plot the g(x) in dashed red
            m1, b1 = -w[1] / w[2], -w[0] / w[2]
            ax.plot(x, m1 * x + b1, 'g--', label='h(x)')
        else:
            # plot the curve g(x)
            x1 = np.linspace(-1, 1)
            x2 = np.linspace(-1, 1)

            X1, X2 = np.meshgrid(x1, x2)

            Y = w[0] + w[1] * X1 + w[2] * X2 + w[3] * X1 * X2 + w[4] * X1 ** 2 + w[5] * X2 ** 2

            ax.contour(X1, X2, Y, [0], colors='g')

        red_patch = patches.Patch(color='red', label='Negative point')
        blue_patch = patches.Patch(color='blue', label='Positive point')
        target_patch = patches.Patch(color='black', label='f(x)')
        g_patch = patches.Patch(color='green', label='g(x)')

        plt.legend(handles=[red_patch, blue_patch, target_patch, g_patch])

        plt.show()

    @staticmethod
    def generate_target_weights():
        # create random line as target function f(x)
        point1 = [1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        point2 = [1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)]

        x1A = point1[1]
        x2A = point1[2]

        x1B = point2[1]
        x2B = point2[2]

        # create the target function weights based on the random points
        target_weights = np.array([x1B * x2A - x1A * x2B, x2B - x1A, x1A - x1B])

        return target_weights

    @staticmethod
    def evaluate_difference(target_labels, hypothesis_labels):
        # evaluate the difference between f and g on in or out-of-sample data.
        sample_size = len(target_labels)

        i = 0
        total_misclassified = 0
        while i < sample_size:
            target_classification = target_labels[i]
            hypothesis_classification = hypothesis_labels[i]
            if target_classification != hypothesis_classification:
                total_misclassified += 1
            i += 1

        return total_misclassified / sample_size


class LinearRegression:
    def __init__(self):
        self.target_weights = Utils.generate_target_weights()

    @staticmethod
    def apply(w, x):
        # apply h(x)
        return sign(np.dot(w, x))

    @staticmethod
    def learn(X, Y):
        # learn from misclassifications
        pseudo_inverse = np.linalg.pinv(X)
        return np.dot(pseudo_inverse, Y)

    @staticmethod
    def get_labels(w, X):
        labels = []
        [labels.append(LinearRegression.apply(w, x)) for x in X]
        return np.array([labels]).T

    @staticmethod
    def get_labels_target(X):
        def target(x):
            return sign(x[1] ** 2 + x[2] ** 2 - 0.6)

        labels = []
        [labels.append(target(x)) for x in X]
        return np.array([labels]).T

    @staticmethod
    def non_linear_transformation(X):
        new_X = np.empty([len(X), 6])
        i = 0
        while i < len(X):
            new_X[i] = np.array((X[i][0], X[i][1], X[i][2], X[i][1] * X[i][2], X[i][1] ** 2, X[i][2] ** 2))
            i += 1
        return new_X

    def run(self, number_of_runs, X, Y, transformation=False):
        ein_total = 0
        eout_total = 0
        best_model = None
        best_eout = None

        if transformation is True:
            # Transform training dataset into space Z
            X = LinearRegression.non_linear_transformation(X)

        i = 0
        while i < number_of_runs:
            # Generate hypothesis weights and labels
            hypothesis_weight = LinearRegression.learn(X, Y).T.flatten()
            hypothesis_labels = LinearRegression.get_labels(hypothesis_weight, X)

            # Calculate Ein total
            ein_total += Utils.evaluate_difference(Y, hypothesis_labels)

            # Generate 1000 out_of_sample points
            X_out = Utils.generate_set(1000)

            # Label the out-of-sample points with target function
            Y_out = LinearRegression.get_labels_target(X_out)

            if transformation is True:
                # Transform dataset into space Z
                X_out = LinearRegression.non_linear_transformation(X_out)

            # Label out_of_sample data set with hypothesis
            hypothesis_labels_out = LinearRegression.get_labels(hypothesis_weight, X_out)

            # Calculate Eout total
            eout = Utils.evaluate_difference(Y_out, hypothesis_labels_out)
            eout_total += eout

            if best_eout is None or eout < best_eout:
                best_eout = eout
                best_model = hypothesis_weight
            i += 1
        Utils.display_figures(number_of_runs, ein_total, eout_total)
        if transformation:
            Utils.plot(X, Y, self.target_weights, best_model, transformation)
        else:
            Utils.plot(X, Y, self.target_weights, best_model)


# Generate data set and labels by using target function x1^2 + x2^2 - 0.6
X = Utils.generate_set(1000)
Y = LinearRegression.get_labels_target(X)

# generate 10% noise in the training set
Y = Utils.generate_noise(Y, 0.1)

model = LinearRegression()

print("1. Linear Regression without transformation")
model.run(1000, X, Y, transformation=False)

print("2. Linear Regression with non-linear transformation")
model.run(1000, X, Y, transformation=True)
