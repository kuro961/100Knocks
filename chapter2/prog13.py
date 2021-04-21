import pandas as pd

c1 = pd.read_table('col1.txt', header=None)
c2 = pd.read_table('col2.txt', header=None)
df = pd.concat([c1, c2], axis=1)
df.to_csv('col1-2.txt', sep='\t', index=False, header=False)