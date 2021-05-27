from prog20 import my_argparse
import re
import pandas as pd

def main():
    args = my_argparse()

    df = pd.read_json(args.file, lines=True)
    uk_text = df[df['title'] == 'イギリス']['text'].values[0]
    uk_texts = uk_text.split('\n')

    ans = {}
    for line in uk_texts:
        info = re.search(r'^\|([^=]+)\s=\s*(.*)', line)
        if info:
            ans[info[1]] = info[2]
    print(ans)

if __name__ == '__main__':
    main()