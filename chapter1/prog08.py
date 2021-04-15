def cipher(str: str):
    return ''.join(chr(219 - ord(char)) if char.islower() else char for char in str)

def main():
    str = "Hi He Lied Because Boron Could Not Oxidize Fluorine."

    enc_str = cipher(str)
    dec_str = cipher(enc_str)
    print("入力文字列:", str)
    print("暗号文:", enc_str)
    print("復号文:", dec_str)

if __name__ == '__main__':
    main()