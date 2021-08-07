from prog60 import my_argparse
from gensim.models import KeyedVectors
import pandas as pd

def main():
    args = my_argparse()
    wv = KeyedVectors.load_word2vec_format(args.word2vec_file, binary=True)

    result = []
    with open(args.test_data, 'r') as f:
        for line in f:
            words = line.split()
            if words[0] != ':':
                neighbor, similarity = wv.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)[0]
                result.append(' '.join(words + [neighbor, str(similarity)]))
            else:
                result.append(' '.join(words))

    df = pd.DataFrame(data=result)
    df.to_csv(args.dst_file,index=False, header=False)

if __name__ == '__main__':
    main()