import argparse
import math
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Split a file into N pieces')
    parser.add_argument('file')
    parser.add_argument('N')
    args = parser.parse_args()

    df = pd.read_table(args.file, header=None)
    N = int(args.N)
    line_n = math.ceil(len(df)/N)
    for i in range(N):
        df.loc[line_n*i:line_n*(i+1)].to_csv(f'ans16-{i}.txt', sep='\t', index=False, header=False)

if __name__ == '__main__':
    main()