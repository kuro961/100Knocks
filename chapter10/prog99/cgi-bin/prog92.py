import subprocess

def FairseqInteractive(path, data_dir, src_file, dst_file, beam_size='5'):
    fairseq_interactive = ['fairseq-interactive',
                           '--path', path,
                           '--beam', beam_size,
                           data_dir
                           ]

    with open(src_file, 'r') as f:
        res_fairseq = subprocess.Popen(fairseq_interactive, stdin=f, stdout=subprocess.PIPE)

    res_grep = subprocess.Popen(['grep', '^H'],stdin=res_fairseq.stdout, stdout=subprocess.PIPE)
    res_fairseq.stdout.close()

    with open(dst_file, 'w') as f:
        res_cut = subprocess.Popen(['cut', '-f3'], stdin=res_grep.stdout, stdout=f)
        res_cut.wait()
    res_grep.stdout.close()

def main():
    path = 'save91/checkpoint10.pt'
    data_dir = 'data91'
    src_file = 'test_spacy.ja'
    dst_file = '92.out'
    FairseqInteractive(path, data_dir, src_file, dst_file)

if __name__ == '__main__':
    main()