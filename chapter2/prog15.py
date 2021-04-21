import sys
import pandas as pd

if len(sys.argv) == 1:
    print('行数Nを指定してください')
else:
    n = int(sys.argv[1])
    df = pd.read_table('popular-names.txt', header=None)
    print(df.tail(n))