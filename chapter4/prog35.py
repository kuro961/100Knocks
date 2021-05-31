from prog30 import my_argparse, read_mecab
import collections

def get_words_freq_list(mecab_file):
    blocks = read_mecab(mecab_file)
    exact_words = []
    for block in blocks:
        for word in block:
            exact_words.append(f"{word['base']},{word['pos']},{word['pos1']}")
    ans = collections.Counter(exact_words)
    ans = ans.most_common()

    return ans

def main():
    args = my_argparse()

    freq_list = get_words_freq_list(args.mecab_file)
    print(freq_list)

if __name__ == '__main__':
    main()