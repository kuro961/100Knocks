import argparse

def my_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mecab_file', default='neko.txt.mecab')
    args = parser.parse_args()

    return args

def parse_mecab(block):
    result = []
    for line in block.split('\n'):
        if line == '':
            break
        (surface, attr) = line.split('\t')
        attr = attr.split(',')
        # surface:表層形, base:基本形, pos:品詞, pos1:品詞細分類1
        line_dict = {
            'surface': surface,
            'base': attr[6],
            'pos': attr[0],
            'pos1': attr[1]
        }
        result.append(line_dict)

    return result

def read_mecab(file_name):
    with open(file_name, 'r') as f:
        blocks = f.read().split('EOS\n')
    blocks = [parse_mecab(block) for block in blocks if block != '']

    return blocks

def main():
    args = my_argparse()

    blocks = read_mecab(args.mecab_file)

    #確認用
    print(blocks[0])
    print(blocks[1])
    print(blocks[2])

if __name__ == '__main__':
    main()