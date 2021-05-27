import argparse
import pandas as pd

def my_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='jawiki-country.json')
    args = parser.parse_args()

    return args

def main():
    args = my_argparse()

    df = pd.read_json(args.file, lines=True)
    print(df[df['title'] == 'イギリス']['text'].values[0])

if __name__ == '__main__':
    main()