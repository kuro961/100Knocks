from prog20 import my_argparse, get_uk_text
import re

def main():
    args = my_argparse()

    uk_text = get_uk_text(args.file, split=False)

    for section in re.findall(r'(=+)\s*(.+?)\s*\1\n', uk_text):
        print(section[1], len(section[0])-1)

if __name__ == '__main__':
    main()