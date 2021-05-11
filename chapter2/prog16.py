import sys
import math
import pandas as pd

def main():
    if len(sys.argv) == 1:
        print('分割数Nを指定してください')
    else:
        n = int(sys.argv[1])
        df = pd.read_table('popular-names.txt', header=None)
        line_n = math.ceil(len(df)/n)
        for i in range(n):
            df.loc[line_n*i:line_n*(i+1)].to_csv(f'ans16-{i}.txt', sep='\t', index=False, header=False)

if __name__ == '__main__':
    main()