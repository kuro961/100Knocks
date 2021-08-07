import subprocess

def FairseqScore(sys_file, ref_file):
    fairseq_score = ['fairseq-score',
                     '--sys', sys_file,
                     '--ref', ref_file
                     ]

    res = subprocess.run(fairseq_score, encoding='utf-8', stdout=subprocess.PIPE)
    return res.stdout

def main():
    sys_file = '92.out'
    ref_file = 'test_spacy.en'
    res = FairseqScore(sys_file, ref_file)
    print(res)

if __name__ == '__main__':
    main()