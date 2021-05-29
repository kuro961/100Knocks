from prog20 import my_argparse, get_uk_text
import re

def remove_mk(text):
    #強調マークアップの除去
    #除去対象:''他との区別'', '''強調''', '''''斜体と強調'''''
    text = re.sub(r"'{5}|'{3}|'{2}", '', text)

    #内部リンク、ファイルの除去
    #除去対象:[[記事名]], [[記事名|表示文字]], [[記事名#節名|表示文字]], [[ファイル:Wikipedia-logo-v2-ja.png|thumb|説明文]]
    # [[]]を消すだけの場合
    text = re.sub(r'\[\[(.*?)]]', r'\1', text)

    #外部リンクの除去
    #除去対象:[http://www.example.org 表示文字], [http://www.example.org], http://www.example.org
    #表示文字のみ残す
    text = re.sub(r'\[?(?:https?://[^\s]*\s?)([^]]*)]?', r'\1', text)

    #Template:Langの除去
    #除去対象:{{lang|言語タグ|文字列}}
    text = re.sub(r'{{lang\|(?:[^|}]+\|)([^|}]+)}}', r'\1', text)

    # HTMLタグの除去
    # 除去対象 <xx> </xx> <xx/>
    text = re.sub(r'<.+?>', '', text)

    return text

def main():
    args = my_argparse()

    uk_texts = get_uk_text(args.file)

    original = {}
    ans = {}
    for line in uk_texts:
        info = re.search(r'^\|([^=]+)\s=\s*(.*)', line)
        if info:
            remove_mk_info = remove_mk(info[2])
            ans[info[1]] = remove_mk_info
            original[info[1]] = info[2]

    print('フィールド名', '除去前', '除去後', sep='\t')
    for o, a in zip(original.items(), ans.items()):
        field_name = o[0]
        print(field_name, o[1], a[1], sep='\t')

if __name__ == '__main__':
    main()