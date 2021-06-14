from prog40 import my_argparse
from prog41 import read_cabocha

def main():
    args = my_argparse()

    blocks = read_cabocha(args.file)
    for block in blocks:
        for chunk in block:
            if chunk.dst != '-1':
                print(''.join([mo.surface for mo in chunk.morphs if mo.pos != '記号']),
                      ''.join([mo.surface for mo in block[int(chunk.dst)].morphs if mo.pos != '記号']),
                      sep='\t')

if __name__ == '__main__':
    main()