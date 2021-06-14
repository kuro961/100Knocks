from prog30 import my_argparse, read_mecab
from prog36 import plot_bar_graph
import collections

def main():
    args = my_argparse()

    blocks = read_mecab(args.mecab_file)
    co_occurrence = []
    for block in blocks:
        words = [word['base'] for word in block]
        if '猫' in words:
            tmp = [w for w in words if w != '猫']
            co_occurrence.extend(tmp)
    ans = collections.Counter(co_occurrence)
    ans = ans.most_common()

    plot_bar_graph(ans)

if __name__ == '__main__':
    main()