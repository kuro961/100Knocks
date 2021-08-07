from prog50 import my_argparse
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix

def main():
    args = my_argparse()

    x_train_vec = pd.read_table(args.train_f)
    y_train = pd.read_table(args.train)['CATEGORY']
    x_test_vec = pd.read_table(args.test_f)
    y_test = pd.read_table(args.test)['CATEGORY']

    clf = pickle.load(open(args.model, 'rb'))

    train_pred = clf.predict(x_train_vec)
    train_cm = confusion_matrix(y_train, train_pred)
    print('学習データ')
    print(train_cm)

    test_pred = clf.predict(x_test_vec)
    test_cm = confusion_matrix(y_test, test_pred)
    print('評価データ')
    print(test_cm)

if __name__ == '__main__':
    main()