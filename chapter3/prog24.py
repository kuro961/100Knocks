from prog20 import my_argparse
import re
import pandas as pd

def main():
    args = my_argparse()

    df = pd.read_json(args.file, lines=True)
    uk_text = df[df['title'] == 'イギリス']['text'].values[0]

    for file in re.findall(r'\[\[ファイル:([^]|]+)(\|.+)\|([^]]*)]]', uk_text):
        print(file[0])

if __name__ == '__main__':
    main()