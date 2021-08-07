from prog50 import my_argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_scores(y_true, y_pred, label):
    # 適合率
    precision = precision_score(y_true, y_pred, average=None, labels=label)
    precision = np.append(precision, precision_score(y_true, y_pred, average='micro'))
    precision = np.append(precision, precision_score(y_true, y_pred, average='macro'))

    # 再現率
    recall = recall_score(y_true, y_pred, average=None, labels=label)
    recall = np.append(recall, recall_score(y_true, y_pred, average='micro'))
    recall = np.append(recall, recall_score(y_true, y_pred, average='macro'))

    # F1スコア
    f1 = f1_score(y_true, y_pred, average=None, labels=label)
    f1 = np.append(f1, f1_score(y_true, y_pred, average='micro'))
    f1 = np.append(f1, f1_score(y_true, y_pred, average='macro'))
    scores = pd.DataFrame({'適合率': precision, '再現率': recall, 'F1スコア': f1},
                          index=np.append(label, ['マイクロ平均', 'マクロ平均']))

    return scores

def main():
    args = my_argparse()

    x_test_vec = pd.read_table(args.test_f)
    y_test = pd.read_table(args.test)['CATEGORY']

    clf = pickle.load(open(args.model, 'rb'))

    test_pred = clf.predict(x_test_vec)

    print(calculate_scores(y_test, test_pred, clf.classes_))

if __name__ == '__main__':
    main()