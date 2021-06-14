from prog40 import my_argparse
from prog41 import read_cabocha

def main():
    args = my_argparse()

    blocks = read_cabocha(args.file)
    with open(args.dst, 'w') as f:
        for block in blocks:
            for chunk in block:
                path = []
                if '名詞' in [mo.pos for mo in chunk.morphs] and int(chunk.dst) != -1:
                    path.append(''.join([mo.surface for mo in chunk.morphs]))
                    current_chunk = chunk
                    next_chunk = block[int(chunk.dst)]
                    while int(current_chunk.dst) != -1:
                        path.append(''.join([mo.surface for mo in next_chunk.morphs]))
                        current_chunk = next_chunk
                        next_chunk = block[int(next_chunk.dst)]
                    print(*path, sep=' -> ', file=f)

if __name__ == '__main__':
    main()