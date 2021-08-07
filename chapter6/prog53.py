from prog50 import my_argparse
import pandas as pd
import pickle

def main():
    args = my_argparse()

    x_test_vec = pd.read_table(args.test_f)

    clf = pickle.load(open(args.model, 'rb'))

    print(clf.classes_)
    test_pred = clf.predict_proba(x_test_vec)
    print(test_pred)

if __name__ == '__main__':
    main()