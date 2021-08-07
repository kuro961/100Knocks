from prog60 import my_argparse
from gensim.models import KeyedVectors

def main():
    args = my_argparse()
    wv = KeyedVectors.load_word2vec_format(args.word2vec_file, binary=True)

    print(wv.similarity('United_States', 'U.S.'))

if __name__ == '__main__':
    main()