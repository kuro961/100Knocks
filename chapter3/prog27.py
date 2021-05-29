from prog20 import my_argparse, get_uk_text
import re

def remove_emphasis(text):
    return re.sub(r"'{5}|'{3}|'{2}", '', text)

def remove_internal_links(text):
    # [[]]を消すだけの場合
    return re.sub(r'\[\[(.*?)]]', r'\1', text)

def main():
    args = my_argparse()

    uk_texts = get_uk_text(args.file)

    ans = {}
    for line in uk_texts:
        info = re.search(r'^\|([^=]+)\s=\s*(.*)', line)
        if info:
            remove_emphasis_info = remove_emphasis(info[2])
            remove_internal_links_info = remove_internal_links(remove_emphasis_info)
            ans[info[1]] = remove_internal_links_info
    print(ans)

if __name__ == '__main__':
    main()