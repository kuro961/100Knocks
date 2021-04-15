def get_char_ngram(str: str, n: int):
    return [str[i:i+n] for i in range(len(str) - n + 1)]

def get_word_ngram(str, n):
    words = str.split()
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def main():
    str = "I am an NLPer"
    print(get_char_ngram(str, 2))
    print(get_word_ngram(str,2))

if __name__ == '__main__':
    main()