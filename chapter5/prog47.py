from prog40 import my_argparse
from prog41 import read_cabocha
from prog46 import get_words

def main():
    args = my_argparse()

    blocks = read_cabocha(args.file)

    with open(args.dst, 'w') as f:
        for block in blocks:
            for chunk in block:
                for mo in chunk.morphs:
                    if mo.pos == '動詞':
                        verb = {'base': mo.base, 'srcs': chunk.srcs}
                        predicates_and_srcs = []
                        for src in verb['srcs']:
                            predicates_and_srcs.extend([{'pred': f"{mo.base}を{verb['base']}",
                                                         'srcs': [j for j in verb['srcs'] + block[src].srcs if j != src]}
                                                        for i, mo in enumerate(block[src].morphs)
                                                        if mo.pos1 == 'サ変接続'
                                                        and i < len(block[src].morphs) - 1
                                                        and block[src].morphs[i+1].pos == '助詞'
                                                        and block[src].morphs[i+1].surface == 'を'])

                        if predicates_and_srcs:
                            for predicate_and_srcs in predicates_and_srcs:
                                direct_objects = []
                                for src in predicate_and_srcs['srcs']:
                                    src_morphs = block[int(src)].morphs
                                    direct_objects.extend([(mo.base, get_words(src_morphs)) for mo in src_morphs if mo.pos == '助詞'])

                                if direct_objects:
                                    direct_objects.sort(key=lambda x: x[0])
                                    f.write(f"{predicate_and_srcs['pred']}\t"
                                            f"{' '.join(d[0] for d in direct_objects)}\t"
                                            f"{' '.join(d[1] for d in direct_objects)}\n")

                        break

if __name__ == '__main__':
    main()