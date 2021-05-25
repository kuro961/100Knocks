import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file1', default='col1.txt')
    parser.add_argument('--src_file2', default='col2.txt')
    parser.add_argument('--dst_file', default='ans13.txt')
    args = parser.parse_args()

    c1 = pd.read_table(args.src_file1, header=None)
    c2 = pd.read_table(args.src_file2, header=None)
    df = pd.concat([c1, c2], axis=1)
    df.to_csv(args.dst_file, sep='\t', index=False, header=False)

if __name__ == '__main__':
    main()