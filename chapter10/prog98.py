# JParaCrawlで学習後、KFTTで再学習
from prog90 import join_token_text
from prog91 import FairseqPreprocess, FairseqTrain
from prog92 import FairseqInteractive
from prog93 import FairseqScore
import sentencepiece as spm
import re
import subprocess
import os
import spacy

def main():
    with open('en-ja/en-ja.bicleaner05.txt') as f:
        data = [x.strip().split('\t') for x in f]
        data = [[x[3], x[2]] for x in data if len(x) == 4]

    with open('jparacrawl.ja', 'w') as ja_f, open('jparacrawl.en', 'w') as en_f:
        for j, e in data:
            print(j, file=ja_f)
            print(e, file=en_f)

    # 日本語側のサブワード化
    sp = spm.SentencePieceProcessor()
    sp.Load('kyoto_ja.model')
    with open('jparacrawl.ja') as src, open('jparacrawl_sub.ja', 'w') as dst:
        for x in src:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)
            x = sp.encode_as_pieces(x)
            x = ' '.join(x)
            print(x, file=dst)

    # 英語側のサブワード化
    apply_codes = ['subword-nmt', 'apply-bpe', '-c', 'kyoto_en.codes']
    with open('jparacrawl.en', 'r') as src, open('jparacrawl_sub.en', 'w') as dst:
        subprocess.run(apply_codes, encoding='utf-8', stdin=src, stdout=dst)

    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    # JParaCrawlで学習
    train_pref = 'jparacrawl_sub'
    valid_pref = 'dev_sub'
    dest_dir = 'data98'
    FairseqPreprocess(train_pref, valid_pref, dest_dir)

    data_dir = 'data98'
    save_dir = 'save98_1'
    max_epoch = '3'
    lr = '1e-4'
    warmup_updates = '4000'
    FairseqTrain(data_dir, save_dir, lr, warmup_updates, max_epoch=max_epoch)

    # KFTTで再学習
    train_pref = 'train_sub'
    valid_pref = 'dev_sub'
    dest_dir = 'data98_2'
    tgtdict = 'data98/dict.en.txt'
    srcdict = 'data98/dict.ja.txt'
    FairseqPreprocess(train_pref, valid_pref, dest_dir, tgtdict=tgtdict, srcdict=srcdict)

    data_dir = dest_dir
    save_dir = 'save98_2'
    restore_file = 'save98_1/checkpoint3.pt'
    FairseqTrain(data_dir, save_dir, restore_file=restore_file)

    path = 'save98_2/checkpoint10.pt'
    data_dir = 'data98_2'
    src_file = 'test_sub.ja'
    dst_file = '98.out'
    FairseqInteractive(path, data_dir, src_file, dst_file)

    # トークナイズをSpacyに合わせる
    nlp = spacy.load('en_core_web_md')
    with open(f'98.out', 'r') as src, open(f'98_tmp.out', 'w') as dst:
        subprocess.run(['sed', '-r', 's/(@@ )|(@@ ?$)//g'], stdin=src, stdout=dst)

    with open(f'98_tmp.out', 'r') as src, open(f'98_spacy.out', 'w') as dst:
        for line in src:
            line = line.strip()
            doc = nlp(line)
            dst.write(''.join([join_token_text(sent) for sent in doc.sents]) + '\n')

    sys_file = f'98_spacy.out'
    ref_file = 'test_spacy.en'
    print(FairseqScore(sys_file, ref_file))

if __name__ == '__main__':
    main()