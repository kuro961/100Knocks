from prog20 import my_argparse
import re
import pandas as pd

def remove_emphasis(text):
    return re.sub(r"'{5}|'{3}|'{2}", '', text)

def main():
    args = my_argparse()

    df = pd.read_json(args.file, lines=True)
    uk_text = df[df['title'] == 'イギリス']['text'].values[0]
    uk_texts = uk_text.split('\n')

    ans = {}
    for line in uk_texts:
        info = re.search(r'^\|([^=]+)\s=\s*(.*)', line)
        if info:
            remove_emphasis_info = remove_emphasis(info[2])
            ans[info[1]] = remove_emphasis_info
    print(ans)

if __name__ == '__main__':
    main()