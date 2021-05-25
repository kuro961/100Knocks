import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('N', help='number of lines')
    args = parser.parse_args()

    df = pd.read_table(args.file, header=None)
    N = int(args.N)
    print(df.tail(N))

if __name__ == '__main__':
    main()