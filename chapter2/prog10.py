import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    df = pd.read_table(args.file, header=None)
    print(len(df))

if __name__ == '__main__':
    main()