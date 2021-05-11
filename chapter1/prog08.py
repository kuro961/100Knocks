def cipher(string: str):
    return ''.join(chr(219 - ord(char)) if char.islower() else char for char in string)

def main():
    string = "Hi He Lied Because Boron Could Not Oxidize Fluorine."

    enc_string = cipher(string)
    dec_string = cipher(enc_string)
    print("入力文字列:", string)
    print("暗号文:", enc_string)
    print("復号文:", dec_string)

if __name__ == '__main__':
    main()