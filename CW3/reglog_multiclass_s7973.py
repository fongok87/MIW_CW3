import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
    
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
class Classifier():

    def __init__(self, lrgd1, lrgd2):
        self.lrgd1 = lrgd1
        self.lrgd2 = lrgd2

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.lrgd1.predict(X) == 1, 0, np.where(self.lrgd2.predict(X) == 1, 2, 1))

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    #w regresji logarytmicznej wyjście przyjmuje wartości 0 lub 1 (prawdopodobieństwa)
    y_train_01_subset = y_train.copy()
    y_train_02_subset = y_train.copy()

    y_train_01_subset[(y_train == 0) | (y_train == 1)] = 1
    y_train_01_subset[(y_train_01_subset == 2)] = 0
    
    y_train_02_subset[(y_train == 2) | (y_train == 0)] = 0
    y_train_02_subset[(y_train_02_subset == 1)] = 1

    lrgd1 = LogisticRegressionGD()
    lrgd2 = LogisticRegressionGD()
    lrgd1.fit(X_train, y_train_01_subset)
    lrgd2.fit(X_train, y_train_02_subset)
    classifier = Classifier(lrgd1, lrgd2)
    #y_1_activation = classifier.activation(y_train_01_subset)
    #y_2_activation = classifier.activation(y_train_02_subset)
    y_3_activation = classifier.activation(X_train)
    #print(y_1_activation)
    #print(y_2_activation)
    print(y_3_activation)
    plot_decision_regions(X_train, y_train, classifier=classifier)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
