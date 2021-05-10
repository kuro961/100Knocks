def get_ngram(target, n):
    return [target[i:i+n] for i in range(len(target) - n + 1)]

def main():
    string = "I am an NLPer"
    print(get_ngram(string, 2))
    print(get_ngram(string.split(), 3))

if __name__ == '__main__':
    main()