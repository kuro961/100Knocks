import pandas as pd

def main():
    df = pd.read_table('popular-names.txt', header=None)
    df[0].value_counts().to_csv('ans19.txt', sep='\t', header=False)

if __name__ == '__main__':
    main()