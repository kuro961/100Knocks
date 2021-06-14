from prog30 import my_argparse, read_mecab

def main():
    args = my_argparse()

    blocks = read_mecab(args.mecab_file)
    ans = []
    for block in blocks:
        for i in range(1, len(block) - 1):
            if (block[i - 1]['pos'] == '名詞'
                    and block[i]['surface'] == 'の'
                    and block[i + 1]['pos'] == '名詞'):
                ans.append((block[i - 1]['surface'], block[i + 1]['surface']))
    print(ans)

if __name__ == '__main__':
    main()