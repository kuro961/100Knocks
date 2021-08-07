from prog91 import FairseqTrain
import os
import subprocess

def main():
    '''
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    data_dir = 'data95'
    save_dir = 'save96'
    tensorboard_logdir = 'log96'
    FairseqTrain(data_dir, save_dir, tensorboard_logdir)
    '''
    tensorboard_logdir = 'log96'
    subprocess.run(['tensorboard', '--logdir', tensorboard_logdir])

    # localhost:6006などを開く

if __name__ == '__main__':
  main()