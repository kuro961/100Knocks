import argparse
import string
import torch
import pandas as pd
from gensim.models import KeyedVectors

def my_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_file', default='GoogleNews-vectors-negative300.bin.gz')
    parser.add_argument('--train', default='train.txt')
    parser.add_argument('--x_train', default='x_train.pt')
    parser.add_argument('--y_train', default='y_train.pt')
    parser.add_argument('--valid', default='valid.txt')
    parser.add_argument('--x_valid', default='x_valid.pt')
    parser.add_argument('--y_valid', default='y_valid.pt')
    parser.add_argument('--test', default='test.txt')
    parser.add_argument('--x_test', default='x_test.pt')
    parser.add_argument('--y_test', default='y_test.pt')
    parser.add_argument('--figure_file', default='figure.png')
    args = parser.parse_args()
    return args

def transform_w2v(text, wv):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    words = text.translate(table).split()
    vec = [wv[word] for word in words if word in wv]

    return torch.tensor(sum(vec) / len(vec))

def main():
    args = my_argparse()

    wv = KeyedVectors.load_word2vec_format(args.word2vec_file, binary=True)

    train_df = pd.read_table(args.train)
    valid_df = pd.read_table(args.valid)
    test_df = pd.read_table(args.test)

    x_train = torch.stack([transform_w2v(line, wv) for line in train_df['TITLE']])
    x_valid = torch.stack([transform_w2v(line, wv) for line in valid_df['TITLE']])
    x_test = torch.stack([transform_w2v(line, wv) for line in test_df['TITLE']])

    category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = torch.tensor(train_df['CATEGORY'].map(lambda x: category_dict[x]).values)
    y_valid = torch.tensor(valid_df['CATEGORY'].map(lambda x: category_dict[x]).values)
    y_test = torch.tensor(test_df['CATEGORY'].map(lambda x: category_dict[x]).values)

    torch.save(x_train, args.x_train)
    torch.save(x_valid, args.x_valid)
    torch.save(x_test, args.x_test)
    torch.save(y_train, args.y_train)
    torch.save(y_valid, args.y_valid)
    torch.save(y_test, args.y_test)

if __name__ == '__main__':
    main()