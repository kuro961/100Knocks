import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def my_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='newsCorpora.csv')
    parser.add_argument('--train', default='train.txt')
    parser.add_argument('--valid', default='valid.txt')
    parser.add_argument('--test', default='test.txt')
    parser.add_argument('--train_f', default='train.feature.txt')
    parser.add_argument('--valid_f', default='valid.feature.txt')
    parser.add_argument('--test_f', default='test.feature.txt')
    parser.add_argument('--model', default='model.sav')
    args = parser.parse_args()

    return args

def main():
    args = my_argparse()

    newsCorpora = pd.read_table(args.file, header=None,
                                names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

    flag = newsCorpora['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])
    newsCorpora = newsCorpora[flag].sample(frac=1, random_state=1)
    x = newsCorpora.loc[:, ['TITLE', 'CATEGORY']]

    x_train, x_valid_test = train_test_split(x, test_size=0.2, random_state=1)
    x_valid, x_test = train_test_split(x_valid_test, test_size=0.5, random_state=1)

    x_train.to_csv(args.train, sep='\t', index=False)
    x_valid.to_csv(args.valid, sep='\t', index=False)
    x_test.to_csv(args.test, sep='\t', index=False)

    # 事例数の確認
    print('学習データ')
    print(x_train['CATEGORY'].value_counts())
    print('検証データ')
    print(x_valid['CATEGORY'].value_counts())
    print('評価データ')
    print(x_test['CATEGORY'].value_counts())

if __name__ == '__main__':
    main()