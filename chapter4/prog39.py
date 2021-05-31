from prog30 import my_argparse
from prog35 import get_words_freq_list
import math
import matplotlib.pyplot as plt

def main():
    args = my_argparse()

    freq_list = get_words_freq_list(args.mecab_file)
    x_data = []
    y_data = []
    for i, d in enumerate(freq_list):
        x_data.append(math.log(i+1))
        y_data.append(math.log(d[1]))

    plt.figure(figsize=(8, 8))
    plt.scatter(x_data, y_data)
    plt.show()

if __name__ == '__main__':
    main()