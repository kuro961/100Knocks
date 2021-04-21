import pandas as pd

df = pd.read_table('popular-names.txt', header=None)
s = set(df[0])
print(s)