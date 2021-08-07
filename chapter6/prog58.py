from prog50 import my_argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    args = my_argparse()

    x_train_vec = pd.read_table(args.train_f)
    y_train = pd.read_table(args.train)['CATEGORY']
    x_valid_vec = pd.read_table(args.valid_f)
    y_valid = pd.read_table(args.valid)['CATEGORY']
    x_test_vec = pd.read_table(args.test_f)
    y_test = pd.read_table(args.test)['CATEGORY']

    result = []
    for C in [10**i for i in range(-5,6)]:
        clf = LogisticRegression(solver='liblinear', C=C)
        clf.fit(x_train_vec, y_train)

        train_pred = clf.predict(x_train_vec)
        valid_pred = clf.predict(x_valid_vec)
        test_pred = clf.predict(x_test_vec)

        train_score = accuracy_score(y_train, train_pred)
        valid_score = accuracy_score(y_valid, valid_pred)
        test_score = accuracy_score(y_test, test_pred)

        result.append([C, train_score, valid_score, test_score])

    result = np.array(result).T
    plt.plot(result[0], result[1], label='train', linestyle="solid")
    plt.plot(result[0], result[2], label='valid', linestyle="dashed")
    plt.plot(result[0], result[3], label='test', linestyle="dashdot")
    plt.legend()
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.xlabel('C')
    plt.xscale('log')
    plt.show()

if __name__ == '__main__':
    main()