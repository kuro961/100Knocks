import random

def get_typoglycemia_text(string: str):
    words = string.split()
    return ' '.join([word[0]+''.join(random.sample(word[1:-1],len(word[1:-1])))+word[-1]
                     if len(word) > 4
                     else word
                     for word in words])

def main():
    string = 'I couldn\'t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'
    print('入力文字列:', string)
    string = get_typoglycemia_text(string)
    print('出力文字列:', string)

if __name__ == '__main__':
    main()