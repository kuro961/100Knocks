from prog30 import my_argparse, read_mecab

def main():
    args = my_argparse()

    blocks = read_mecab(args.mecab_file)
    ans = []
    for block in blocks:
        tmp_ans = []
        for word in block:
            if word['pos'] == '名詞':
                tmp_ans.append(word['surface'])
            else:
                if len(tmp_ans) >= 2:
                    ans.append(tmp_ans)
                tmp_ans = []
    print(ans)

if __name__ == '__main__':
    main()