import pandas as pd

df = pd.read_table('popular-names.txt', header=None)
df_s = df.sort_values(2, ascending=False)
df_s.to_csv(f'ans18.txt', sep='\t', index=False, header=False)