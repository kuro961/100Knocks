from prog90 import join_token_text
from prog91 import FairseqPreprocess, FairseqTrain
from prog92 import FairseqInteractive
from prog93 import FairseqScore
from prog94 import plot_scores, wait_child_process
import os
import re
import sys
import spacy
import subprocess
import sentencepiece as spm

def main():
    # 日本語側のサブワード化
    spm.SentencePieceTrainer.Train('--input=kftt-data-1.0/data/orig/kyoto-train.ja --model_prefix=kyoto_ja --vocab_size=16000 --character_coverage=1.0')
    sp = spm.SentencePieceProcessor()
    sp.Load('kyoto_ja.model')

    for src, dst in [
        ('kftt-data-1.0/data/orig/kyoto-train.ja', 'train_sub.ja'),
        ('kftt-data-1.0/data/orig/kyoto-dev.ja', 'dev_sub.ja'),
        ('kftt-data-1.0/data/orig/kyoto-test.ja', 'test_sub.ja'),
    ]:
        with open(src) as f, open(dst, 'w') as g:
            for x in f:
                x = x.strip()
                x = re.sub(r'\s+', ' ', x)
                x = sp.encode_as_pieces(x)
                x = ' '.join(x)
                print(x, file=g)

    # 英語側のサブワード化
    learn_codes = ['subword-nmt', 'learn-bpe', '-s', '16000']
    with open('kftt-data-1.0/data/orig/kyoto-train.en', 'r') as s, open('kyoto_en.codes', 'w') as d:
        subprocess.run(learn_codes, encoding='utf-8',stdin=s, stdout=d)

    apply_codes = ['subword-nmt', 'apply-bpe', '-c', 'kyoto_en.codes']
    for src, dst in [
        ('kftt-data-1.0/data/orig/kyoto-train.en', 'train_sub.en'),
        ('kftt-data-1.0/data/orig/kyoto-dev.en', 'dev_sub.en'),
        ('kftt-data-1.0/data/orig/kyoto-test.en', 'test_sub.en'),
    ]:
        with open(src, 'r') as s, open(dst, 'w') as d:
            subprocess.run(apply_codes, encoding='utf-8',stdin=s, stdout=d)

    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    train_pref = 'train_sub'
    valid_pref = 'dev_sub'
    dest_dir = 'data95'
    FairseqPreprocess(train_pref, valid_pref, dest_dir)

    data_dir = dest_dir
    save_dir = 'save95'
    FairseqTrain(data_dir, save_dir)

    max_beam_size = 40
    step = 3

    path = 'save95/checkpoint10.pt'
    data_dir = 'data95'
    src_file = 'test_sub.ja'
    for i in range(1, max_beam_size + 1, step):
        dst_file = f'95.{i}.out'
        beam_size = str(i)
        pid = os.fork()
        if pid == 0:
            FairseqInteractive(path, data_dir, src_file, dst_file, beam_size)
            sys.exit()
    wait_child_process()

    # トークナイズをSpacyに合わせる
    nlp = spacy.load('en_core_web_md')
    for i in range(1, max_beam_size + 1, step):
        with open(f'95.{i}.out', 'r') as src, open(f'95.{i}_tmp.out', 'w') as dst:
            subprocess.run(['sed', '-r', 's/(@@ )|(@@ ?$)//g'], stdin=src, stdout=dst)

        with open(f'95.{i}_tmp.out', 'r') as src, open(f'95.{i}_spacy.out', 'w') as dst:
            for line in src:
                line = line.strip()
                doc = nlp(line)
                dst.write(''.join([join_token_text(sent) for sent in doc.sents]) + '\n')

    scores_file = '95.scores'
    with open(scores_file, 'w') as f:
        for i in range(1, max_beam_size + 1, step):
            sys_file = f'95.{i}_spacy.out'
            ref_file = 'test_spacy.en'
            f.write(FairseqScore(sys_file, ref_file))

    plot_scores(scores_file, max_beam_size, step)

if __name__ == '__main__':
    main()