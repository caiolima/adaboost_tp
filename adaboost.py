import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_hastie_10_2

class AdaBoost:

    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.h = []
        self.alpha = []

    def fit(self, X_train, Y_train):
        n_train = len(X_train)

        # Init weights
        w = np.ones(n_train) / n_train
        self.h = []
        self.alpha = []
        for i in range(self.n_estimators):
            clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
            # Fit a classifier with the specific weights
            clf_tree.fit(X_train, Y_train, sample_weight = w)
            self.h.append(clf_tree)
            pred_train_i = clf_tree.predict(X_train)
            # Indicator function
            miss = [int(x) for x in (pred_train_i != Y_train)]
            # Equivalent with 1/-1 to update weights
            miss2 = [x if x==1 else -1 for x in miss]
            # Error
            err_m = np.dot(w,miss) / sum(w)
            # Alpha
            alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
            self.alpha.append(alpha_m)

            # New weights
            w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))

    def predict(self, X):
        pred_test = np.zeros(len(X))
        for i in range(self.n_estimators):
            y_pred = self.h[i].predict(X)
            pred_test = [sum(x) for x in zip(pred_test, 
                                         [x * self.alpha[i] for x in y_pred])]

        return np.sign(pred_test)

x, y = make_hastie_10_2()
df = pd.DataFrame(x)
df['Y'] = y
train, test = train_test_split(df, test_size = 0.2)
X_train, Y_train = train.ix[:,:-1], train.ix[:,-1]
X_test, Y_test = test.ix[:,:-1], test.ix[:,-1]

clf = AdaBoost(100)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
y_test = Y_test.values

accuracy = 0.0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        accuracy += 1.0 / float(len(y_pred))

print(accuracy)

