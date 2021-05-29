from prog20 import my_argparse, get_uk_text
import re

def main():
    args = my_argparse()

    uk_text = get_uk_text(args.file, split=False)

    for file in re.findall(r'\[\[ファイル:(.*?)\|', uk_text):
        print(file)

if __name__ == '__main__':
    main()