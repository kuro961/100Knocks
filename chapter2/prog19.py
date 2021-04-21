import pandas as pd

df = pd.read_table('popular-names.txt', header=None)
df[0].value_counts().to_csv(f'ans19.txt', sep='\t', header=False)