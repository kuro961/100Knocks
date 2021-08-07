from prog60 import my_argparse
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def main():
    args = my_argparse()
    wv = KeyedVectors.load_word2vec_format(args.word2vec_file, binary=True)

    countries = set()
    is_countries_1_3 = False
    is_countries_0_2 = False
    with open(args.test_data, 'r') as f:
        for line in f:
            words = line.split()
            if words[0] == ':':
                if words[1] in ['capital-common-countries', 'capital-world']:
                    is_countries_1_3 = True
                    is_countries_0_2 = False
                elif words[1] in ['currency', 'gram6-nationality-adjective']:
                    is_countries_0_2 = True
                    is_countries_1_3 = False
                else:
                    is_countries_1_3 = False
                    is_countries_0_2 = False
            else:
                if is_countries_1_3:
                    countries.add(words[1])
                    countries.add(words[3])
                elif is_countries_0_2:
                    countries.add(words[0])
                    countries.add(words[2])
    countries = list(countries)

    countries_vec = np.array([wv[country] for country in countries])

    plt.figure(figsize=(20, 10))
    z = linkage(countries_vec, method='ward')
    dendrogram(z, labels=countries)
    plt.show()
    plt.savefig(args.figure_file)

if __name__ == '__main__':
    main()