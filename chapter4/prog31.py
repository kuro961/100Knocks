from prog30 import my_argparse, read_mecab

def main():
    args = my_argparse()

    blocks = read_mecab(args.mecab_file)
    ans = []
    for block in blocks:
        for word in block:
            if word['pos'] == '動詞':
                ans.append(word['surface'])
    print(ans)

if __name__ == '__main__':
    main()