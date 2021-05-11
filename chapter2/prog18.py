import pandas as pd

def main():
    df = pd.read_table('popular-names.txt', header=None)
    df_s = df.sort_values(2, ascending=False)
    df_s.to_csv('ans18.txt', sep='\t', index=False, header=False)

if __name__ == '__main__':
    main()