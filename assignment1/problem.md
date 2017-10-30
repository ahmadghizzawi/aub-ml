In this problem, you will create your own target function *f* and data set *D* to see how the Perceptron Learning Algorithm works. Take *d = 2* so you can visualize the problem, and assume *X* = [−1, 1] × [−1, 1] with uniform probability of picking each *x* ∈ *X*.
<br>
<br>
In each run, choose a random line in the plane as your target function *f* (do this by taking two random, uniformly distributed points in [−1, 1] × [−1, 1] and taking the line passing through them), where one side of the line maps to +1 and the other maps to −1. Choose the inputs *x_n* of the data set as random points (uniformly in *X*), and evaluate the target function on each *x_n* to get the corresponding output *y_n*.
<br>
<br>
Now, in each run, use the Perceptron Learning Algorithm to find *g*. Start the PLA with the weight vector *w* being all zeros (consider *sign(0) = 0*, so all points are initially misclassified), and at each iteration have the algorithm choose a point randomly from the set of misclassified points. We are interested in two quantities: the number of iterations that PLA takes to converge to *f*, and the disagreement between *f* and *g* which is *P[f(x)* ≠ *g(x)]* (i.e., the probability that *f* and *g* will disagree on their classification of a random point). You can either calculate this probability exactly, or approximate it by generating a sufficiently large, separate set of points to estimate it.
<br>
<br>
In order to get a reliable estimate for these two quantities, you should repeat the experiment for 1000 runs (each run as specified above) and take the average over these runs.
<br>
<br>

Report the below four figures(numbers) as well as a visualization of your best Perceptron among all 1000 runs for *N = 10* and *N = 100* training points (one visualization for each case). The best Perceptron is the one with the least *P[f(x) ≠ g(x)]*. Your visualization should include the training points, the line representing the target function *f* and the line representing your final hypothesis *g*.
