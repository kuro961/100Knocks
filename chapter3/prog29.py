from prog20 import my_argparse
import re
import requests
import pandas as pd

def main():
    args = my_argparse()

    df = pd.read_json(args.file, lines=True)
    uk_text = df[df['title'] == 'イギリス']['text'].values[0]
    uk_texts = uk_text.split('\n')

    ans = {}
    for line in uk_texts:
        info = re.search(r'^\|([^=]+)\s=\s*(.*)', line)
        if info:
            ans[info[1]] = info[2]
    url_national_flag = ans['国旗画像']
    URL = 'https://www.mediawiki.org/w/api.php'
    PARAMS = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url",
        "titles": "File:" + url_national_flag,
    }
    R = requests.get(url=URL, params=PARAMS)
    DATA = R.json()
    urls = DATA['query']['pages']['-1']['imageinfo'][0]
    url = urls['url']
    print(url)

if __name__ == '__main__':
    main()