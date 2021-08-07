from prog92 import FairseqInteractive
from prog93 import FairseqScore
import os
import re
import sys
import matplotlib.pyplot as plt

def plot_scores(filename, max_beam_size, step):
    scores = []
    with open(filename, 'r') as f:
        for line in f:
            score = re.search(r'(?<=BLEU4 = )\d*\.\d*(?=,)', line)
            if score is not None:
                scores.append(float(score.group()))

    beam_size = range(1, max_beam_size + 1, step)
    plt.plot(beam_size, scores)
    plt.show()

def wait_child_process():
    while True:
        try:
            os.wait()
        except ChildProcessError:
            return

def main():
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    max_beam_size = 40
    step = 3

    path = 'save91/checkpoint10.pt'
    data_dir = 'data91'
    src_file = 'test_spacy.ja'
    for i in range(1, max_beam_size + 1, step):
        dst_file = f'94.{i}.out'
        beam_size = str(i)
        pid = os.fork()
        if pid == 0:
            FairseqInteractive(path, data_dir, src_file, dst_file, beam_size)
            sys.exit()
    wait_child_process()

    scores_file = '94.scores'
    with open(scores_file, 'w') as f:
        for i in range(1, max_beam_size + 1, step):
            sys_file = f'94.{i}.out'
            ref_file = 'test_spacy.en'
            f.write(FairseqScore(sys_file, ref_file))

    plot_scores(scores_file, max_beam_size, step)

if __name__ == '__main__':
    main()