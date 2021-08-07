#!/home/kuroda/anaconda3/bin/python3.8

from prog92 import FairseqInteractive
from jinja2 import Template
import sentencepiece as spm
import subprocess
import re
import cgi
import sys
import io
import os

html = '''
<!DOCTYPE html>
<html lang="ja">
<head>
    <title>My Translate</title>
    <meta charset="UTF-8">
</head>
<body>
    <form method="post" action="prog99_2.py">
        <textarea name="ja" cols="40">{{ja}}</textarea>
        <textarea name="en" cols="40">{{en}}</textarea><br>        
        <input type="submit" value="翻訳"/>
    </form>
</body>
</html>
'''
template = Template(html)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
form = cgi.FieldStorage()
if 'ja' not in form:
    ja = ''
    en = ''
else:
    ja = form.getvalue('ja')
    with open('99.ja', 'w') as f:
        f.write(ja)

    sp = spm.SentencePieceProcessor()
    sp.Load('/home/kuroda/100Knocks/chapter10/kyoto_ja.model')
    with open('99.ja') as src, open('99_sub.ja', 'w') as dst:
        for x in src:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)
            x = sp.encode_as_pieces(x)
            x = ' '.join(x)
            print(x, file=dst)

    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    path = '/home/kuroda/100Knocks/chapter10/save98_2/checkpoint10.pt'
    data_dir = '/home/kuroda/100Knocks/chapter10/data98_2'
    src_file = '99_sub.ja'
    dst_file = '99.out'
    FairseqInteractive(path, data_dir, src_file, dst_file)

    with open(f'99.out', 'r') as src:
        restored = subprocess.run(['sed', '-r', 's/(@@ )|(@@ ?$)//g'], stdin=src, stdout=subprocess.PIPE)

    en = restored.stdout.decode('utf8')

data = {'en': en, 'ja': ja}
print(template.render(data))