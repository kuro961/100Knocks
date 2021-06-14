from prog40 import my_argparse
from prog41 import read_cabocha

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
                            direct_objects.extend([mo.base for mo in src_morphs if mo.pos == '助詞'])

                        if direct_objects:
                            direct_objects = ' '.join(sorted(direct_objects))
                            f.write(f"{predicate}\t{direct_objects}\n")

                        break

if __name__ == '__main__':
    main()

# UNUXコマンド
# 頻出する述語と格パターンの組み合わせ
#sort ans.txt | uniq --count | sort --numeric-sort --reverse

# それぞれの動詞の格パターンの出現頻度が高い順
# 「行う」の場合
#grep "^行う\s" ans.txt | sort | uniq --count | sort --numeric-sort --reverse

# 「なる」の場合
#grep "^なる\s" ans.txt | sort | uniq --count | sort --numeric-sort --reverse

# 「与える」の場合
#grep "^与える\s" ans.txt | sort | uniq --count | sort --numeric-sort --reverse