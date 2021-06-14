from prog40 import my_argparse
from prog41 import read_cabocha

def is_include_noun(morphs):
    for mo in morphs:
        if mo.pos == '名詞':
            return True
    return False

def is_include_verb(morphs):
    for mo in morphs:
        if mo.pos == '動詞':
            return True
    return False

def main():
    args = my_argparse()

    blocks = read_cabocha(args.file)
    for block in blocks:
        for chunk in block:
            if chunk.dst != '-1':
                src_morphs = chunk.morphs
                dst_morphs = block[int(chunk.dst)].morphs
                if is_include_noun(src_morphs) and is_include_verb(dst_morphs):
                    print(''.join([mo.surface for mo in src_morphs if mo.pos != '記号']),
                          ''.join([mo.surface for mo in dst_morphs if mo.pos != '記号']),
                          sep='\t')

if __name__ == '__main__':
    main()