string = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
words = string.split()
word_dict = {}

for i, word in enumerate(words):
    if i + 1 in [1, 5, 6, 7, 8, 9, 15, 16, 19]:
        word_dict[word[:1]] = i + 1
    else:
        word_dict[word[:2]] = i + 1

print(word_dict)