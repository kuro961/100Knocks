from prog50 import my_argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

def main():
    args = my_argparse()

    x_train_vec = pd.read_table(args.train_f)
    y_train = pd.read_table(args.train)['CATEGORY']

    clf = LogisticRegression(solver='liblinear')
    clf.fit(x_train_vec, y_train)

    pickle.dump(clf, open(args.model, 'wb'))

if __name__ == '__main__':
    main()