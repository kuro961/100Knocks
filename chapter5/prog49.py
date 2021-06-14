from prog40 import my_argparse
from prog41 import read_cabocha

def get_path(block, chunk):
    path = []
    current_chunk = chunk
    next_chunk = block[int(chunk.dst)]
    while int(current_chunk.dst) != -1:
        path.append(current_chunk.dst)
        current_chunk = next_chunk
        next_chunk = block[int(next_chunk.dst)]
    return path

def main():
    args = my_argparse()

    blocks = read_cabocha(args.file)

    with open(args.dst, 'w') as f:
        for block in blocks:
            noun_phrases = [chunk for chunk in block if '名詞' in [mo.pos for mo in chunk.morphs]]
            for i in range(len(noun_phrases) - 1):
                str1 = ''.join([mo.surface if mo.pos != '名詞' else 'X' for mo in noun_phrases[i].morphs])
                if int(noun_phrases[i].dst) != -1:
                    str1_path = get_path(block, noun_phrases[i])
                    for j in range(i+1, len(noun_phrases)):
                        str2 = ''.join([mo.surface if mo.pos != '名詞' else 'Y' for mo in noun_phrases[j].morphs])
                        str2_path = get_path(block, noun_phrases[j])
                        #文節iから構文木の根に至る経路上に文節jが存在する場合
                        if str(j) in str1_path:
                            path = [str1]
                            for p in str1_path:
                                if p == str(j):
                                    break
                                path.append(''.join([mo.surface for mo in block[int(p)].morphs]))
                            path.append(str2)
                            print(*path, sep=' -> ', file=f)
                        #文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合
                        else:
                            k = list(set(str1_path) & set(str2_path))
                            if len(k) > 0:
                                for m in k:
                                    str1_to_k = [str1]
                                    for p in str1_path:
                                        if p == m:
                                            break
                                        str1_to_k.append(''.join([mo.surface for mo in block[int(p)].morphs]))
                                    print(*str1_to_k, sep=' -> ', end='', file=f)
                                    print(' | ', end='', file=f)
                                    str2_to_k = [str2]
                                    for p in str2_path:
                                        if p == m:
                                            break
                                        str2_to_k.append(''.join([mo.surface for mo in block[int(p)].morphs]))
                                    print(*str2_to_k, sep=' -> ', end='', file=f)
                                    print(' | ', end='', file=f)
                                    print(''.join([mo.surface for mo in block[int(m)].morphs]), file=f)

if __name__ == '__main__':
    main()