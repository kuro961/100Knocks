from prog40 import my_argparse
from prog41 import read_cabocha
from graphviz import Digraph

def main():
    args = my_argparse()

    graph = Digraph(format='png')
    blocks = read_cabocha(args.file)
    # 例として2ブロック目の係り受け木を作成
    target = blocks[1]
    for i, chunk in enumerate(target):
        graph.node(str(i), ''.join([mo.surface for mo in chunk.morphs if mo.pos != '記号']))

    for i, chunk in enumerate(target):
        if chunk.dst != '-1':
            graph.edge(str(i), chunk.dst)
    graph.render(args.png)

if __name__ == '__main__':
    main()