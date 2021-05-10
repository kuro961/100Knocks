import re

def main():
    string = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    string = re.sub('[,.]', '', string)
    len_list = [len(word) for word in string.split()]
    print(len_list)

if __name__ == '__main__':
    main()