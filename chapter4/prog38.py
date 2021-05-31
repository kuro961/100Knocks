from prog30 import my_argparse
from prog35 import get_words_freq_list
import matplotlib.pyplot as plt

def main():
    args = my_argparse()

    freq_list = get_words_freq_list(args.mecab_file)

    hist_data = []
    for word in freq_list:
        hist_data.append(word[1])

    plt.hist(hist_data, bins=100)
    plt.show()

if __name__ == '__main__':
    main()