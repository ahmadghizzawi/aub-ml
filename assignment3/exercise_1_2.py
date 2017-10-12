import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Utils:
    @staticmethod
    def generate_set(N, d):
        d = d - 1
        x0 = np.ones((N, 1))

        if d != 0:
            # Generate random uniformally distributes points
            training_set = np.random.uniform(-1, 1, size=(N, d))
            # Insert x0 values into the begining of the array.
            training_set = np.insert(training_set, [0], x0, axis=1)
        else:
            training_set = x0

        return training_set

    @staticmethod
    def display_figures(num_of_runs, ein_total, eout_total):
        print('Average Ein(g): ' + str(ein_total / num_of_runs))
        print('Average Eout(g): ' + str(eout_total / num_of_runs))

    @staticmethod
    def plot(w_0, w_1, regularized_w):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        # change the axis to fit our dataset
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        x = np.linspace(-1, 1)

        y = np.sin(np.pi * x)

        y_0 = w_0[0] * x
        y_1 = w_1[0] * x + w_1[1] * x
        y_regularized = regularized_w[0] * x + regularized_w[1] * x

        ax.plot(x, y, 'g')

        ax.plot(x, y_0, 'r')

        ax.plot(x, y_1, 'b')

        ax.plot(x, y_regularized, 'y--')

        red_patch = patches.Patch(color='r', label='Average g0')
        blue_patch = patches.Patch(color='b', label='Average g1')
        target_patch = patches.Patch(color='y', label='Average regularized g1')
        g_patch = patches.Patch(color='green', label='f(x)')

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
    def __init__(self, d, regularization=None):
        self.target_weights = Utils.generate_target_weights()

        self.regularization = regularization

        # Generate 1000 out_of_sample points
        self.X_out = Utils.generate_set(1000, d)

        if d < 2:
            X = Utils.generate_set(1000, 2)
            self.Y_out = LinearRegression.get_labels_sin(X)
        else:
            # Label out-of-sample points
            self.Y_out = LinearRegression.get_labels_sin(self.X_out)

        self.d = d

    @staticmethod
    def apply(w, x):
        # apply h(x)
        return np.dot(w, x)

    @staticmethod
    def learn(X, Y):
        # learn from misclassifications
        pseudo_inverse = np.linalg.pinv(X)
        return np.dot(pseudo_inverse, Y).T.flatten()

    @staticmethod
    def regularized_learn(X, Y, reg_var):
        # learn from misclassifications
        term_X = np.linalg.inv(np.dot(X.T, X) + reg_var * np.eye(len(X)))

        pseudo_inverse_regularized = np.dot(term_X, X.T)

        return np.dot(pseudo_inverse_regularized, Y).T.flatten()

    @staticmethod
    def get_labels(w, X):
        labels = []
        [labels.append(LinearRegression.apply(w, x)) for x in X]
        return np.array([labels]).T

    @staticmethod
    def get_labels_sin(X):
        def target(x):
            return np.sin(np.pi * x[1])

        labels = []
        [labels.append(target(x)) for x in X]
        return np.array([labels]).T

    def expected_g_bar(self, average_hypothesis):
        sum_g_bar = 0
        i = 0
        for x in self.X_out:
            g_bar = LinearRegression.apply(average_hypothesis, x)
            sum_g_bar += g_bar
            i += 1
        return sum_g_bar / len(self.X_out)

    def expected_bias(self, average_hypothesis):
        sum_bias = 0
        i = 0
        for x in self.X_out:
            g_bar = LinearRegression.apply(average_hypothesis, x)
            sum_bias += (g_bar - self.Y_out[i, 0])**2
            i += 1
        return sum_bias/len(self.X_out)

    def expected_variance(self, all_hypotheses, average_hypothesis):
        sum_variance = 0
        i = 0
        for x in self.X_out:
            g_bar = LinearRegression.apply(average_hypothesis, x)
            g = LinearRegression.apply(all_hypotheses[i], x)
            sum_variance += (g - g_bar) ** 2
            i += 1
        return sum_variance / len(self.X_out)

    def run(self, training_sets=1000):
        # Returns g_average
        d = self.d
        # hypotheses set for h0
        all_hypotheses = np.zeros((1000, d))

        i = 0
        while i < training_sets:
            # Generate data set
            X = Utils.generate_set(2, d)

            if d < 2:
                # Labels by using target function sin(pi * x)
                X_temp = Utils.generate_set(2, 2)
                Y = LinearRegression.get_labels_sin(X_temp)
            else:
                # Labels by using target function sin(pi * x)
                Y = LinearRegression.get_labels_sin(X)

            if self.regularization:
                w = LinearRegression.regularized_learn(X, Y, self.regularization)
            else:
                w = LinearRegression.learn(X, Y)

            all_hypotheses[i] = w

            i += 1

        average_g = np.mean(all_hypotheses, axis=0)

        # Expected bias for hypotheses
        bias = self.expected_bias(average_g)

        # Expected variance for hypotheses
        variance = self.expected_variance(all_hypotheses, average_g)

        # calculate expected value of the average hypothesis
        print("Expected gbar(x): " + str(self.expected_g_bar(average_g)))

        print("Bias: " + str(bias))

        print("Variance: " + str(variance))

        # Expected out-of-sample error
        print("Out-of-sample Error: " + str(bias + variance))

        return average_g

print("1. h0(x) = w0")
model = LinearRegression(d=1)
h0_average_hypothesis = model.run(1000)

print('----')

print("2. h1(x) = w0 + x1w1")
model = LinearRegression(d=2)
h1_average_hypothesis = model.run(1000)

print('----')

print("Regularized h1(x) lambda = 0")
model = LinearRegression(d=2, regularization=0)
model.run(1000)

print('----')

print("Regularized h1(x) lambda = 0.001")
model = LinearRegression(d=2, regularization=0.001)
model.run(1000)

print('----')

print("Regularized h1(x) lambda = 0.01")
model = LinearRegression(d=2, regularization=0.01)
model.run(1000)

print('----')

print("Regularized h1(x) lambda = 0.1")
model = LinearRegression(d=2, regularization=0.1)
model.run(1000)

print('----')

print("Regularized h1(x) lambda = 1")
model = LinearRegression(d=2, regularization=1)
regularized_average_hypothesis = model.run(1000)

print('----')

print("Regularized h1(x) lambda = 10")
model = LinearRegression(d=2, regularization=10)
model.run(1000)

Utils.plot(h0_average_hypothesis, h1_average_hypothesis, regularized_average_hypothesis)
