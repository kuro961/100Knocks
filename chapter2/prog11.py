import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', default='popular-names.txt')
    parser.add_argument('--dst_file', default='ans11.txt')
    args = parser.parse_args()

    df = pd.read_table(args.src_file, header=None)
    df.to_csv(args.dst_file, sep=' ', index=False, header=False)

if __name__ == '__main__':
    main()