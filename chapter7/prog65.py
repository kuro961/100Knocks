from prog60 import my_argparse

def main():
    args = my_argparse()

    with open(args.dst_file, 'r') as f:
        sem_cnt = sem_cor = 0
        syn_cnt = syn_cor = 0
        for line in f:
            words = line.split()
            if line.startswith(': gram'):
               category = 'syntactic'
            elif line.startswith(':'):
                category = 'semantic'
            else:
                if category == 'semantic':
                    sem_cnt += 1
                    if words[3] == words[4]:
                        sem_cor += 1
                else:
                    syn_cnt += 1
                    if words[3] == words[4]:
                        syn_cor += 1

    print(f'意味的アナロジー : {sem_cor/sem_cnt:.3f}')
    print(f'文法的アナロジー : {syn_cor/syn_cnt:.3f}')

if __name__ == '__main__':
    main()