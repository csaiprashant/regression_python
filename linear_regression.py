import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('food_truck_profits.txt', header=None, names=['x1', 'x2'])
X = np.array(df['x1'])
y = np.array(df['x2'])
X = X.reshape(X.size, 1)
y = y.reshape(y.size, 1)

m, n = X.shape
X = np.hstack((np.ones((m, 1)), X))
theta = np.zeros((n + 1, 1))

MAX_ITER = 1500
ALPHA = 0.01


def compute_cost(X, theta, y):
    m = np.size(y)
    h = np.dot(X, theta)
    squared_error = np.square(h - y)
    J = 0.5 * np.sum(squared_error) / m
    return J


J_history = []
cost = compute_cost(X, theta, y)
print('Initial Cost = {}'.format(cost))
J_history.append(cost)
for i in range(1, MAX_ITER):
    h = np.dot(X, theta)
    theta = theta - ALPHA * np.dot(X.T, h - y) / m
    cost = compute_cost(X, theta, y)
    print('Cost after {}th iteration = {}'.format(i, cost))
    J_history.append(cost)

print('Predicted 3500-sq feet house price = {}'.format(np.dot(np.array([1, 3500]), theta)))
print('theta = {}'.format(theta))

plt.plot(J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.show()

y_hat = np.dot(X, theta)
df['x3'] = y_hat
plt.scatter(df['x1'], df['x2'])
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(df['x1'], df['x3'], color='orange')
plt.show()
