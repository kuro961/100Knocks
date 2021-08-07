import os
import subprocess

def FairseqPreprocess(train_pref, valid_pref, dest_dir, thresholdsrc='0',thresholdtgt='0', tgtdict=None, srcdict=None):
    fairseq_preprocess = ['fairseq-preprocess', '-s', 'ja', '-t', 'en',
                          '--trainpref', train_pref,
                          '--validpref', valid_pref,
                          '--destdir', dest_dir,
                          '--thresholdsrc', thresholdsrc,
                          '--thresholdtgt', thresholdtgt,
                          '--workers', '20'
                          ]
    if tgtdict is not None:
        fairseq_preprocess.extend(['--tgtdict', tgtdict])
    if srcdict is not None:
        fairseq_preprocess.extend(['--srcdict', srcdict])

    subprocess.run(fairseq_preprocess)

def FairseqTrain(data_dir, save_dir, lr='1e-3', dropout='0.1', warmup_updates='2000', tensorboard_logdir='no tensorboard logging', max_epoch='10', restore_file='checkpoint_last.pt'):
    fairseq_train = ['fairseq-train', data_dir,
                     '--fp16',
                     '--restore-file', restore_file,
                     '--tensorboard-logdir', tensorboard_logdir,
                     '--save-dir', save_dir,
                     '--max-epoch', max_epoch,
                     '--arch', 'transformer',
                     '--share-decoder-input-output-embed',
                     '--optimizer', 'adam',
                     '--clip-norm', '1.0',
                     '--lr', lr,
                     '--lr-scheduler', 'inverse_sqrt',
                     '--warmup-updates', warmup_updates,
                     '--update-freq', '1',
                     '--dropout', dropout,
                     '--weight-decay', '0.0001',
                     '--criterion', 'label_smoothed_cross_entropy',
                     '--label-smoothing', '0.1',
                     '--max-tokens', '8000'
                     ]
    subprocess.run(fairseq_train)

def main():
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    train_pref = 'train_spacy'
    valid_pref = 'dev_spacy'
    dest_dir = 'data91'
    thresholdsrc = '5'
    thresholdtgt = '5'
    FairseqPreprocess(train_pref, valid_pref, dest_dir, thresholdsrc, thresholdtgt)

    data_dir = dest_dir
    save_dir = 'save91'
    FairseqTrain(data_dir, save_dir)

if __name__ == '__main__':
    main()