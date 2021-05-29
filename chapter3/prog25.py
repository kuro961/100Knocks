from prog20 import my_argparse, get_uk_text
import re

def main():
    args = my_argparse()

    uk_texts = get_uk_text(args.file)

    ans = {}
    for line in uk_texts:
        info = re.search(r'^\|([^=]+)\s=\s*(.*)', line)
        if info:
            ans[info[1]] = info[2]
            print(ans)

if __name__ == '__main__':
    main()