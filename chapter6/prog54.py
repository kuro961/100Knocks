from prog50 import my_argparse
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

def main():
    args = my_argparse()

    x_train_vec = pd.read_table(args.train_f)
    y_train = pd.read_table(args.train)['CATEGORY']
    x_test_vec = pd.read_table(args.test_f)
    y_test = pd.read_table(args.test)['CATEGORY']

    clf = pickle.load(open(args.model, 'rb'))

    train_pred = clf.predict(x_train_vec)
    train_score = accuracy_score(y_train, train_pred)

    test_pred = clf.predict(x_test_vec)
    test_score = accuracy_score(y_test, test_pred)

    print(f'正解率(学習データ) : {train_score:.3f}')
    print(f'正解率(評価データ) : {test_score:.3f}')

if __name__ == '__main__':
    main()