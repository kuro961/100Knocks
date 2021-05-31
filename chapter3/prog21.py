from prog20 import my_argparse, get_uk_text

def main():
    args = my_argparse()

    uk_texts = get_uk_text(args.file)

    ans = list(filter(lambda x: '[Category:' in x, uk_texts))
    print(ans)

if __name__ == '__main__':
    main()