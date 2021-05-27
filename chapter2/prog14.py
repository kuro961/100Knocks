import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='popular-names.txt')
    parser.add_argument('-N', '--number', default=10, type=int, help='number of lines')
    args = parser.parse_args()

    df = pd.read_table(args.file, header=None)
    print(df.head(args.number))

if __name__ == '__main__':
    main()