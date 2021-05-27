from prog20 import my_argparse
import pandas as pd

def main():
    args = my_argparse()

    df = pd.read_json(args.file, lines=True)
    uk_text = df[df['title'] == 'イギリス']['text'].values[0]

    uk_texts = uk_text.split('\n')
    ans = list(filter(lambda x: '[Category:' in x, uk_texts))
    print(ans)

if __name__ == '__main__':
    main()