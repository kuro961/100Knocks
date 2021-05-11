import pandas as pd

def mian():
    df = pd.read_table('popular-names.txt', header=None)
    df[0].to_csv('col1.txt', index=False, header=False)
    df[1].to_csv('col2.txt', index=False, header=False)

if __name__ == '__main__':
    mian()