from prog90 import join_token_text
from prog91 import FairseqTrain
from prog92 import FairseqInteractive
from prog93 import FairseqScore
import os
import re
import spacy
import subprocess

def main():
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    lr_list = ['1e-4', '1e-3']
    dropout_list = ['0.1', '0.2', '0.3', '0.4']
    for lr in lr_list:
        for dropout in dropout_list:
            data_dir = 'data95'
            save_dir = f'save97_{lr}_{dropout}'
            FairseqTrain(data_dir, save_dir, lr=lr, dropout=dropout)

            path = f'{save_dir}/checkpoint10.pt'
            data_dir = 'data95'
            src_file = 'test_sub.ja'
            dst_file = f'97_{lr}_{dropout}.out'
            FairseqInteractive(path, data_dir, src_file, dst_file)

    # トークナイズをSpacyに合わせる
    nlp = spacy.load('en_core_web_md')
    for lr in lr_list:
        for dropout in dropout_list:
            with open(f'97_{lr}_{dropout}.out', 'r') as src, open(f'97_{lr}_{dropout}_tmp.out', 'w') as dst:
                subprocess.run(['sed', '-r', 's/(@@ )|(@@ ?$)//g'], stdin=src, stdout=dst)

            with open(f'97_{lr}_{dropout}_tmp.out', 'r') as src, open(f'97_{lr}_{dropout}_spacy.out', 'w') as dst:
                for line in src:
                    line = line.strip()
                    doc = nlp(line)
                    dst.write(''.join([join_token_text(sent) for sent in doc.sents]) + '\n')

    scores_file = '97.scores'
    with open(scores_file, 'w') as f:
        for lr in lr_list:
            for dropout in dropout_list:
                sys_file = f'97_{lr}_{dropout}_spacy.out'
                ref_file = 'test_spacy.en'
                f.write(FairseqScore(sys_file, ref_file))

    scores = []
    with open(scores_file, 'r') as f:
        for line in f:
            score = re.search(r'(?<=BLEU4 = )\d*\.\d*(?=,)', line)
            if score is not None:
                scores.append(float(score.group()))
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    lr_i = sorted_scores[0][0] // len(dropout_list)
    dropout_i = sorted_scores[0][0] % len(dropout_list)
    print(f'BEST SCORE: BLUE4={sorted_scores[0][1]}')
    print(f'lr: {lr_list[lr_i]}, dropout: {dropout_list[dropout_i]}')

if __name__ == '__main__':
    main()