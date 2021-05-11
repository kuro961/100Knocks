import pandas as pd

def main():
    df = pd.read_table('popular-names.txt', header=None)
    print(len(df))

if __name__ == '__main__':
    main()