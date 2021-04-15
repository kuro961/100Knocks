str1 = "パトカー"
str2 = "タクシー"

str3 = ""
for char1, char2 in zip(str1, str2):
    str3 += char1 + char2

print(str3)