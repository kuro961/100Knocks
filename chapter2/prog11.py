with open('new-popular-names.txt', 'w') as f1:
    with open('popular-names.txt', 'r') as f2:
        for line in f2:
            line = line.replace('\t', ' ')
            f1.write(line)