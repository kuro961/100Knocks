import argparse
from gensim.models import KeyedVectors

def my_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_file', default='GoogleNews-vectors-negative300.bin.gz')
    parser.add_argument('--test_data', default='questions-words.txt')
    parser.add_argument('--test_collection', default='combined.csv')
    parser.add_argument('--dst_file', default='ans.txt')
    parser.add_argument('--figure_file', default='figure.png')
    args = parser.parse_args()
    return args

def main():
    args = my_argparse()
    wv = KeyedVectors.load_word2vec_format(args.word2vec_file, binary=True)

    print(wv['United_States'])

if __name__ == '__main__':
    main()