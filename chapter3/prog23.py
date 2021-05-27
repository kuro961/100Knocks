from prog20 import my_argparse
import re
import pandas as pd

def main():
    args = my_argparse()

    df = pd.read_json(args.file, lines=True)
    uk_text = df[df['title'] == 'イギリス']['text'].values[0]

    for section in re.findall(r'(=+)([^=]+)\1\n', uk_text):
        print(section[1], len(section[0])-1)

if __name__ == '__main__':
    main()