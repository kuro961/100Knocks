import pandas as pd

def main():
    df = pd.read_table('popular-names.txt', header=None)
    df.to_csv('ans11.txt', sep=' ', index=False, header=False)

if __name__ == '__main__':
    main()