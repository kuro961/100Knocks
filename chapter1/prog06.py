def get_char_ngram(str: str, n: int):
    return [str[i:i+n] for i in range(len(str) - n + 1)]

def main():
    str1 = "paraparaparadise"
    str2 = "paragraph"

    str1_char_bigram = set(get_char_ngram(str1, 2))
    str2_char_bigram = set(get_char_ngram(str2, 2))
    print("X:", str1_char_bigram, "\nY:", str2_char_bigram)
    print("和集合:", str1_char_bigram | str2_char_bigram)
    print("積集合:", str1_char_bigram & str2_char_bigram)
    print("差集合:", str1_char_bigram - str2_char_bigram)
    print("Xにseが含まれるか:", "se" in str1_char_bigram)
    print("Yにseが含まれるか:", "se" in str2_char_bigram)

if __name__ == '__main__':
    main()