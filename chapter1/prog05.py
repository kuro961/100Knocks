def get_char_ngram(string: str, n: int):
    return [string[i:i+n] for i in range(len(string) - n + 1)]

def get_word_ngram(string, n):
    words = string.split()
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def main():
    string = "I am an NLPer"
    print(get_char_ngram(string, 2))
    print(get_word_ngram(string,2))

if __name__ == '__main__':
    main()