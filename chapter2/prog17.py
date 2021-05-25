import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    df = pd.read_table(args.file, header=None)
    s = set(df[0])
    print(s)

if __name__ == '__main__':
    main()