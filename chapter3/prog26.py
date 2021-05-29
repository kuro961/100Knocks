from prog20 import my_argparse, get_uk_text
import re

def remove_emphasis(text):
    return re.sub(r"'{5}|'{3}|'{2}", '', text)

def main():
    args = my_argparse()

    uk_texts = get_uk_text(args.file)

    ans = {}
    for line in uk_texts:
        info = re.search(r'^\|([^=]+)\s=\s*(.*)', line)
        if info:
            remove_emphasis_info = remove_emphasis(info[2])
            ans[info[1]] = remove_emphasis_info
    print(ans)

if __name__ == '__main__':
    main()