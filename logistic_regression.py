import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('university_admissions.txt', header=None, names=['x1', 'x2', 'y'])

X = np.array(df[['x1', 'x2']])
y = np.array(df['y'])
y = y.reshape(y.size, 1)
X = np.hstack((np.ones((y.size, 1)), X))
theta = np.zeros((X.shape[1], 1))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(X, y, theta):
    h = sigmoid(np.dot(X, theta))
    cost = y * np.log(h) + (1 - y) * np.log(1 - h)
    return -cost.mean()


cost1 = cost_function(X, y, theta)
cost_list = [cost1]
m, n = X.shape
while True:
    h = sigmoid(np.dot(X, theta))
    gradient = np.dot(X.T, (h - y)) / y.size
    diag = np.multiply(h, (1 - h)) * np.identity(m)
    hessian = (1 / m) * np.dot(np.dot(X.T, diag), X)
    theta = theta - np.dot(np.linalg.inv(hessian), gradient)
    cost2 = cost_function(X, y, theta)
    cost_list.append(cost2)
    if cost2 >= cost1:
        break
    else:
        cost1 = cost2
    print(cost2)


plt.plot(cost_list)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.show()

pos = plt.scatter(df[df['y'] == 1]['x1'], df[df['y'] == 1]['x2'], color='red')
neg = plt.scatter(df[df['y'] == 0]['x1'], df[df['y'] == 0]['x2'], color='blue')
plt.legend((pos, neg), ('Admitted', 'Not Admitted'))
ax = plt.gca()
x1_val = np.array(ax.get_xlim())
x2_val = - (x1_val * theta[1] + theta[0]) / theta[2]
plt.plot(x1_val, x2_val, color='black')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()
