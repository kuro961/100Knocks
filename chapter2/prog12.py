import pandas as pd

df = pd.read_table('popular-names.txt', header=None)
with open('col1.txt', 'w') as f:
    col1 = '\n'.join(df[0])
    f.write(col1)
with open('col2.txt', 'w') as f:
    col2 = '\n'.join(df[1])
    f.write(col2)