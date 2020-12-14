import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import naive_bayes
from sklearn.model_selection import train_test_split

def fit(x, y, n_dimensions):
    n_features = x.shape[1]
    class_labels = np.unique(y)

    # Standardising data
    mean_overall = np.mean(x, axis=0)

    # Covariance matrices
    sw = np.zeros((n_features, n_features))
    sb = np.zeros((n_features, n_features))

    for c in class_labels:
        x_c = x[y == c]
        # racunamo srednju vrednost feature-a koji odgovaraju jednoj klasi
        mean_c = np.mean(x_c, axis=0)
        # po formuli racunamo unutar-klasnu kovarijacionu matricu
        sw += (x_c - mean_c).T.dot((x_c - mean_c))

        n_c = x_c.shape[0]
        diff = (mean_c - mean_overall).reshape(n_features, 1)
        # po formuli racunamo medjuklasnu kovarijacionu matricu
        sb += n_c * diff.dot(diff.T)

    a = np.linalg.inv(sw).dot(sb)
    eigenvalues, eigenvectors = np.linalg.eig(a)
    eigenvectors = eigenvectors.T
    indexes = np.argsort(abs(eigenvalues))[::-1]
    eigenvectors = eigenvectors[indexes]
    linear_disc = eigenvectors[0:n_dimensions]
    return linear_disc


def transform(x, linear_discriminant):
    # smanjujemo dimenzionalnost feature-a
    return np.dot(x, linear_discriminant.T)


df = pd.read_csv("multiclass_data.csv", header=None)

X = df.iloc[:, 0:5].values

Y = df.iloc[:, -1].values

classes = np.unique(Y)

linear_discriminants = fit(X, Y, 2)
X_reduced = transform(X, linear_discriminants)

# Using reduced feature matrix as input into another classifier
X_train, X_test, Y_train, Y_test = train_test_split(X_reduced, Y, train_size=0.95, shuffle=False)

means, variances, priors = naive_bayes.fit(X_train, Y_train, classes)

y_prediction = naive_bayes.predict(X_test, classes, means, variances, priors)

naive_bayes.accuracy(Y_test, y_prediction)

x1 = X_reduced[:, 0]
x2 = X_reduced[:, 1]

# Plotting 2D data
plt.scatter(x1, x2,
            c=Y, edgecolor='none', alpha=0.8,
            cmap=plt.cm.get_cmap('viridis', 3))

plt.xlabel('Linearna diskriminanta 1')
plt.ylabel('Linearna diskriminanta 2')
plt.show()

