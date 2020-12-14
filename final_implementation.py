from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt

df = pd.read_csv("data.csv", header=None)

X = df.iloc[:, 0:5].values
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

Y = df.iloc[:, -1].values
Y = (Y - np.mean(Y)) / np.std(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.95, shuffle=False)

# Procena stepena polinoma za regresiju metodom krosvalidacije

number_of_folds = 5
step = int(X_train.shape[0] / number_of_folds)
degrees = np.arange(1, 10)
alphas = np.arange(0, 1, 0.01)
mean_squared_error_test = []
mean_squared_error_train = []
Y_mean = np.mean(Y_train)

for i in range(0, number_of_folds):
    test = X_train[int(i*step):int((i+1)*step), :]
    train = np.delete(X_train, slice(int(i*step), int((i+1)*step)), 0)
    for j in degrees:
        poly_features = PolynomialFeatures(degree=j, include_bias=True)
        train_poly = poly_features.fit_transform(train)
        test_poly = poly_features.fit_transform(test)
        transposed = np.transpose(train_poly)
        first_factor = np.linalg.inv(transposed.dot(train_poly))
        second_factor = transposed.dot(np.delete(Y_train, slice(int(i*step), int((i+1)*step)), 0))
        theta_optimal = first_factor.dot(second_factor)
        mean_squared_error_train.append((1 / (train_poly.shape[0])) * (np.square(np.linalg.norm(train_poly.dot(theta_optimal) - np.delete(Y_train, slice(int(i*step), int((i+1)*step)), 0)))))
        mean_squared_error_test.append((1/(test_poly.shape[0]))*(np.square(np.linalg.norm(test_poly.dot(theta_optimal) - Y_train[int(i*step):int((i+1)*step)]))))

mean_squared_error_train = np.reshape(mean_squared_error_train, (9, 5))
mean_squared_error_test = np.reshape(mean_squared_error_test, (9, 5))

train_scores_mean = np.mean(mean_squared_error_train, axis=1)
train_scores_std = np.std(mean_squared_error_train, axis=1)
test_scores_mean = np.mean(mean_squared_error_test, axis=1)
test_scores_std = np.std(mean_squared_error_test, axis=1)

plt.title("Validaciona kriva")
plt.xlabel("stepen polinoma")
plt.ylabel("MSE")
lw = 2
plt.plot(degrees, train_scores_mean, label="Trening",
            color="darkorange")
plt.fill_between(degrees, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange")
plt.plot(degrees, test_scores_mean, label="Validacija",
             color="navy")
plt.fill_between(degrees, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy")
plt.legend(loc="best")
plt.show()

# Rezultat ovog dela nagovestava da treba koristiti stepen 3, sledi regularizacija

mean_squared_error_test_regularisation = []

for i in range(0, number_of_folds):
    test = X_train[int(i*step):int((i+1)*step), :]
    train = np.delete(X_train, slice(int(i*step), int((i+1)*step)), 0)
    for a in alphas:
        poly_features = PolynomialFeatures(degree=3, include_bias=False)
        train_poly = poly_features.fit_transform(train)
        test_poly = poly_features.fit_transform(test)
        train_transposed = np.transpose(train_poly)
        first_product = train_transposed.dot(train_poly)
        second_product = np.linalg.inv(first_product + a * np.identity(first_product.shape[0]))
        third_product = train_transposed.dot(np.delete(Y_train, slice(int(i*step), int((i+1)*step)), 0))
        theta_optimal = second_product.dot(third_product)
        theta_optimal = np.insert(theta_optimal, 0, np.array(Y_mean))
        test_poly = np.c_[np.ones(test_poly.shape[0]), test_poly]
        mean_squared_error_test_regularisation.append((1/(test_poly.shape[0]))*(np.square(np.linalg.norm(test_poly.dot(theta_optimal) - Y_train[int(i*step):int((i+1)*step)]))))

mean_squared_error_test_regularisation = np.reshape(mean_squared_error_test_regularisation, (100, 5))

test_scores_mean = np.mean(mean_squared_error_test_regularisation, axis=1)
test_scores_std = np.std(mean_squared_error_test_regularisation, axis=1)

plt.title("Validaciona kriva")
plt.xlabel("Vrednost lambda")
plt.ylabel("MSE")
lw = 2
plt.plot(alphas, test_scores_mean, label="Validacija",
             color="navy")
plt.fill_between(alphas, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy")
plt.legend(loc="best")
plt.show()

# Rezultat izvrsavanja ovog dela nagovestava da se moze izabrati alpha odnosno lambda = 0.1

poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X_test)
alpha = 0.9

X_transposed = np.transpose(X_poly)

first_product = X_transposed.dot(X_poly)

second_product = np.linalg.inv(first_product + alpha * np.identity(first_product.shape[0]))

third_product = X_transposed.dot(Y_test)

theta_optimal = second_product.dot(third_product)

theta_optimal = np.insert(theta_optimal, 0, np.array(Y_mean))

X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]

Y_pred = X_poly.dot(theta_optimal)

plotpoints = np.arange(0, X_test.shape[0])

plt.plot(plotpoints, Y_pred, label='final')
plt.plot(plotpoints, Y_test, label='istina')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()

mean_squared_error = (1/(X_poly.shape[0]))*(np.square(np.linalg.norm(Y_pred - Y_test)))

print(mean_squared_error)
