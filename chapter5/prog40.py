import argparse

def my_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='ai.ja.txt.parsed')
    parser.add_argument('--png', default='ans')
    parser.add_argument('--dst', default='ans.txt')
    args = parser.parse_args()

    return args

class Morph:
    def __init__(self, dc):
        self.surface = dc['surface']
        self.base = dc['base']
        self.pos = dc['pos']
        self.pos1 = dc['pos1']

def parse_cabocha(block):
    res = []
    for line in block.split('\n'):
        if line == '':
            break
        elif line[0] == '*':
            continue
        (surface, attr) = line.split('\t')
        attr = attr.split(',')
        line_dict = {
            'surface': surface,
            'base': attr[6],
            'pos': attr[0],
            'pos1': attr[1]
        }
        res.append(Morph(line_dict))

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
    print([vars(w) for w in blocks[0]])
    print([vars(w) for w in blocks[1]])
    print([vars(w) for w in blocks[2]])

if __name__ == '__main__':
    main()