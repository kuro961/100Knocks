from prog40 import my_argparse
from prog41 import read_cabocha

def get_words(morphs):
    return ''.join([mo.surface for mo in morphs if mo.pos != '記号'])

def main():
    args = my_argparse()

    blocks = read_cabocha(args.file)
    with open(args.dst, 'w') as f:
        for block in blocks:
            for chunk in block:
                for mo in chunk.morphs:
                    if mo.pos == '動詞':
                        verb = {'base': mo.base, 'srcs': chunk.srcs}
                        predicate = verb['base']
                        direct_objects = []
                        for src in verb['srcs']:
                            src_morphs = block[src].morphs
                            direct_objects.extend([(mo.base, get_words(src_morphs)) for mo in src_morphs if mo.pos == '助詞'])

                        if direct_objects:
                            direct_objects.sort(key=lambda x: x[0])
                            f.write(f"{predicate}\t{' '.join(d[0] for d in direct_objects)}\t{' '.join(d[1] for d in direct_objects)}\n")

                        break

if __name__ == '__main__':
    main()