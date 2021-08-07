from prog60 import my_argparse
from gensim.models import KeyedVectors
import pandas as pd

def main():
    args = my_argparse()
    wv = KeyedVectors.load_word2vec_format(args.word2vec_file, binary=True)
    df = pd.read_csv(args.test_collection)

    similarity = []
    for w1, w2 in zip(df['Word 1'], df['Word 2']):
        similarity.append(wv.similarity(w1, w2))

    mean_df = pd.DataFrame({'Human (mean)': df['Human (mean)'], 'Machine (mean)': similarity})
    print(mean_df.corr(method='spearman'))

if __name__ == '__main__':
    main()