import argparse
import pandas as pd
import string
import torch

def my_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train.txt')
    parser.add_argument('--valid', default='valid.txt')
    parser.add_argument('--test', default='test.txt')
    parser.add_argument('--src_word2id', default='train.txt')
    parser.add_argument('--x_train', default='x_train.pt')
    parser.add_argument('--y_train', default='y_train.pt')
    parser.add_argument('--x_valid', default='x_valid.pt')
    parser.add_argument('--y_valid', default='y_valid.pt')
    parser.add_argument('--x_test', default='x_test.pt')
    parser.add_argument('--y_test', default='y_test.pt')
    args = parser.parse_args()

    return args

def get_word2id(src):
    train = pd.read_table(src)

    train_dict = {}
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    for text in train['TITLE']:
        for word in text.translate(table).split():
            train_dict[word] = train_dict.get(word, 0) + 1

    train_dict = sorted(train_dict.items(), key=lambda x:x[1], reverse=True)

    word2id = {word: i + 1 for i, (word, cnt) in enumerate(train_dict) if cnt > 1}

    return word2id

def tokenizer(text, word2id, unknown=0):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return [word2id.get(word, unknown) for word in text.translate(table).split()]

def main():
    args = my_argparse()

    train_df = pd.read_table(args.train)
    valid_df = pd.read_table(args.valid)
    test_df = pd.read_table(args.test)

    word2id = get_word2id(args.train)
    x_train = [torch.tensor(tokenizer(line, word2id)) for line in train_df['TITLE']]
    x_valid = [torch.tensor(tokenizer(line, word2id)) for line in valid_df['TITLE']]
    x_test = [torch.tensor(tokenizer(line, word2id)) for line in test_df['TITLE']]

    category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = torch.tensor(train_df['CATEGORY'].map(lambda x: category_dict[x]).values)
    y_valid = torch.tensor(valid_df['CATEGORY'].map(lambda x: category_dict[x]).values)
    y_test = torch.tensor(test_df['CATEGORY'].map(lambda x: category_dict[x]).values)

    #　後の問題で使用
    torch.save(x_train, 'x_train.pt')
    torch.save(x_valid, 'x_valid.pt')
    torch.save(x_test, 'x_test.pt')
    torch.save(y_train, 'y_train.pt')
    torch.save(y_valid, 'y_valid.pt')
    torch.save(y_test, 'y_test.pt')

if __name__ == '__main__':
    main()