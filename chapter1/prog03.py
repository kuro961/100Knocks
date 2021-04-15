import re

str = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

print([len(re.sub('[,.]', '', word)) for word in str.split()])