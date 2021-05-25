import argparse
import pandas as pd

def mian():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', default='popular-names.txt')
    parser.add_argument('--dst_file1', default='col1.txt')
    parser.add_argument('--dst_file2', default='col2.txt')
    args = parser.parse_args()

    df = pd.read_table(args.src_file, header=None)
    df[0].to_csv(args.dst_file1, index=False, header=False)
    df[1].to_csv(args.dst_file2, index=False, header=False)

if __name__ == '__main__':
    mian()