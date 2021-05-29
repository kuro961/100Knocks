import argparse
import pandas as pd

def my_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='jawiki-country.json.gz')
    args = parser.parse_args()

    return args

def get_uk_text(file, split=True):
    df = pd.read_json(file, lines=True)
    uk_text = df[df['title'] == 'イギリス']['text'].values[0]
    if split:
        uk_text = uk_text.split('\n')

    return uk_text

def main():
    args = my_argparse()

    uk_text = get_uk_text(args.file, split=False)
    print(uk_text)

if __name__ == '__main__':
    main()