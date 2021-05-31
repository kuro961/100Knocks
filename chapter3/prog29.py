from prog20 import my_argparse, get_uk_text
import re
import requests

URL = 'https://www.mediawiki.org/w/api.php'

def main():
    args = my_argparse()

    uk_texts = get_uk_text(args.file)

    url_national_flag = {}
    for line in uk_texts:
        info = re.search(r'^\|([^=]+)\s=\s*(.*)', line)
        if info and info[1] == '国旗画像':
            url_national_flag = info[2]
    PARAMS = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url",
        "titles": "File:" + url_national_flag
    }
    R = requests.get(url=URL, params=PARAMS)
    DATA = R.json()
    urls = DATA['query']['pages']['-1']['imageinfo'][0]
    url = urls['url']
    print(url)

if __name__ == '__main__':
    main()