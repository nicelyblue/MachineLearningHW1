import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def loss_function(x, theta):
    lin_mod = x.dot(theta)
    first_factor = np.sum(lin_mod)
    second_factor = np.log(np.sum(np.exp(lin_mod)))

    loss = first_factor - second_factor

    return loss


def predict(x, theta):
    lin_mod = np.float64(x.dot(theta))
    y_predicted = np.float64(np.exp(lin_mod) / np.sum(np.exp(lin_mod)))
    return y_predicted


def fit(learning_rate, num_iters, batch_size, x, y):
    theta = np.zeros((x.shape[1], 3))
    loss = []
    for i in range(num_iters):
        indexes = np.random.randint(0, len(x), batch_size)
        x = np.take(x, indexes, axis=0)
        y = np.take(y, indexes, axis=0)
        n = x.shape[0]
        for j in range(0, batch_size):
            y_predicted = predict(x, theta)
            gradient = np.dot(x.T, (y - y_predicted))
            gradient = gradient / n
            change = learning_rate * gradient
            theta = theta + change
            theta[:, -1] = 0
        loss.append(loss_function(x, theta))

    return theta, loss


def predict_class(x, theta):
    y_predicted = np.zeros((x.shape[0], 3))
    for i in range(0, x.shape[0]):
        y_predicted[i, :] = predict(x[i, :], theta)
    print(y_predicted)
    y_predicted_cls = np.argmax(y_predicted, axis=1)

    return np.array(y_predicted_cls)


def one_hot_encoder(y):
    shape = (y.size, y.max() + 1)
    one_hot = np.zeros(shape)
    rows = np.arange(y.size)
    one_hot[rows, y] = 1
    return one_hot


def accuracy(y1, y2):
    w = 0
    for i in range(0, len(y1)):
        if y1[i] == y2[i]:
            w += 1

    accuracy = w / len(y1) * 100

    print('The model is ' + str(accuracy) + '% accurate')

    return accuracy


df = pd.read_csv("multiclass_data.csv", header=None)

X = df.iloc[:, 0:5].values
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X = np.c_[np.ones(X.shape[0]), X]

Y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.95, shuffle=True)

Y_train = one_hot_encoder(Y_train)

num_iters = 1000
learning_rate = 0.001

_, loss1 = fit(learning_rate, num_iters, 32, X_train, Y_train)
_, loss2 = fit(learning_rate, num_iters, 64, X_train, Y_train)
_, loss3 = fit(learning_rate, num_iters, 128, X_train, Y_train)

preciznost = []
for i in range(0, 20):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.95, shuffle=True)
    Y_train = one_hot_encoder(Y_train)
    theta, loss = fit(learning_rate, num_iters, 64, X_train, Y_train)
    y_predicted_class = predict_class(X_test, theta)
    preciznost.append(accuracy(y_predicted_class, Y_test))

print(np.mean(preciznost))
plot_points = range(0, num_iters)
plt.plot(plot_points, loss1, 'r', label='batch size = 32')
plt.plot(plot_points, loss2, 'b', label='batch size = 64')
plt.plot(plot_points, loss3, 'g', label='batch size = 128')
plt.legend()
plt.show()
