from prog30 import my_argparse
from prog35 import get_words_freq_list
import matplotlib.pyplot as plt

def plot_bar_graph(data, num=10, figsize=(15,8)):
    label = []
    value = []
    for i in range(num):
        label.append(data[i][0])
        value.append(data[i][1])
    plt.figure(figsize=figsize)
    plt.barh(label, value)
    plt.show()

def main():
    args = my_argparse()

    freq_list = get_words_freq_list(args.mecab_file)

    plot_bar_graph(freq_list)

if __name__ == '__main__':
    main()