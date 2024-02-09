import re
######################################################################################################
# Part 1
######################################################################################################
str = '### Subject: scuba diving at… From: steve.millman@asu.edu Body: Underwater, where it is at'

match = re.search(r':.*:',str)

print("1. ", match.group())

match = re.search(r'st\S*' ,str)

print("2. ", match.group())

match = re.search(r':.*?:' ,str)

print("3. ", match.group())

match = re.search(r'[\w ]*at$',str)

print("4. ", match.group())


######################################################################################################
# Part 2
######################################################################################################

str = 'yoo hoo <<Here is the short text> boom baby to include this for the long stuff> a little extra'

match = re.search(r'<.*?>',str)

print("1. ", match.group())

match = re.search(r'<.*>',str)

print("1. ", match.group())

str = 'okay here is the last example. trying to find MOD22X in the middle of this string'

match = re.search(('MOD'+'([\d]+?)'+'X'),str)

print("1. ", match.group(1))