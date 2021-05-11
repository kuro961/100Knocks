import pandas as pd

def main():
    df = pd.read_table('popular-names.txt', header=None)
    s = set(df[0])
    print(s)

if __name__ == '__main__':
    main()