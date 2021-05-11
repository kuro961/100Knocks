import sys
import pandas as pd

def main():
    if len(sys.argv) == 1:
        print('行数Nを指定してください')
    else:
        n = int(sys.argv[1])
        df = pd.read_table('popular-names.txt', header=None)
        print(df.head(n))

if __name__ == '__main__':
    main()