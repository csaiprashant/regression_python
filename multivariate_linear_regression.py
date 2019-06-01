import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('housing_prices.txt', header=None, names=['x1', 'x2', 'x3'])

df2 = df.copy()

X = df2[['x1', 'x2']].apply(lambda x: (x - np.mean(x)) / np.std(x))
X = np.array(X)
y = np.array(df2['x3'])
y = y.reshape(y.size, 1)
X = np.hstack((np.ones((y.size, 1)), X))
m, n = X.shape
theta = np.zeros((n, 1))


def compute_cost(X, theta, y):
    h = np.dot(X, theta)
    squared_error = np.square(h - y)
    return 0.5 * np.sum(squared_error) / m


MAX_ITER = 1500
ALPHA = 0.01
J_history = []
cost = compute_cost(X, theta, y)
J_history.append(cost)
print('Initial Cost = {}'.format(cost))
for i in range(1, MAX_ITER):
    error = np.dot(X, theta) - y
    theta = theta - ALPHA * np.dot(X.T, error) / m
    cost = compute_cost(X, theta, y)
    print('Cost after {}th iteration = {}'.format(i, cost))
    J_history.append(cost)

print('theta = {}'.format(theta))

plt.plot(J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.show()
