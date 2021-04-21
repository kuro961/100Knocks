import re

string = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

print([len(re.sub('[,.]', '', word)) for word in string.split()])