from prog50 import my_argparse
import pandas as pd
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

    #パラメータ
    param_penalty = ['l1', 'l2']
    param_C = [10**i for i in range(-5,6)]
    #max_iter
    best_score = 0
    best_parameters = {}
    best_clf = LogisticRegression()
    for penalty in param_penalty:
        for C in param_C:
            clf = LogisticRegression(solver='liblinear', penalty=penalty, C=C)
            clf.fit(x_train_vec, y_train)

            valid_pred = clf.predict(x_valid_vec)
            valid_score = accuracy_score(y_valid, valid_pred)

            if valid_score > best_score:
                best_score = valid_score
                best_parameters = {'penalty': penalty, 'C': C}
                best_clf = clf

    test_pred = best_clf.predict(x_test_vec)
    test_score = accuracy_score(y_test, test_pred)

    print(best_parameters)
    print('検証データ{:.4f}'.format(best_score))
    print('評価データ{:.4f}'.format(test_score))

if __name__ == '__main__':
    main()