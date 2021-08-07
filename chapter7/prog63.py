from prog60 import my_argparse
from gensim.models import KeyedVectors

def main():
    args = my_argparse()
    wv = KeyedVectors.load_word2vec_format(args.word2vec_file, binary=True)

    neighbors = wv.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)
    for neighbor in neighbors:
        print(neighbor)

if __name__ == '__main__':
    main()