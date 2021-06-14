from prog40 import Morph, my_argparse

class Chunk:
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []

def parse_cabocha(block):
    res = []
    tmp = []
    dst = None
    for line in block.split('\n'):
        if line == '':
            if tmp:
                chunk = Chunk(tmp, dst)
                res.append(chunk)
                tmp = []
        elif line[0] == '*':
            if tmp:
                chunk = Chunk(tmp, dst)
                res.append(chunk)
                tmp = []
            dst = line.split(' ')[2].rstrip('D')
        else:
            (surface, attr) = line.split('\t')
            attr = attr.split(',')
            line_dict = {
                'surface': surface,
                'base': attr[6],
                'pos': attr[0],
                'pos1': attr[1]
            }
            tmp.append(Morph(line_dict))

    for i, r in enumerate(res):
        res[int(r.dst)].srcs.append(i)

    return res

def read_cabocha(file_name):
    with open(file_name, 'r') as f:
        blocks = f.read().split('EOS\n')
    blocks = [parse_cabocha(block) for block in blocks if block != '']
    return blocks

def main():
    args = my_argparse()

    blocks = read_cabocha(args.file)

    # 確認用
    for chunk in blocks[0]:
        print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)
    print()
    for chunk in blocks[1]:
        print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)

if __name__ == '__main__':
    main()