from prog50 import my_argparse
import pandas as pd
import numpy as np
import pickle

def main():
    args = my_argparse()

    x_train_vec = pd.read_table(args.train_f)
    clf = pickle.load(open(args.model, 'rb'))

    features = x_train_vec.columns.values
    for c, coef in zip(clf.classes_, clf.coef_):
        print(f'[カテゴリ] {c}')
        df = pd.DataFrame(data=[features[np.argsort(coef)[::-1][:10]], features[np.argsort(coef)[:10]]],
                          columns=[i for i in range(1,11)],
                          index=['重みの高い特徴量', '重みの低い特徴量'])
        pd.set_option('display.max_columns', None)
        print(df)

if __name__ == '__main__':
    main()