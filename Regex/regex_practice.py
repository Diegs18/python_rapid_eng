import re           # bring in the package

# define a sample string

my_str = 'Cinderella met a fella at the ball. They all played on the XS57B but not Cinderella, who played on the S23X.'

# find the longest string starting and ending with ll
long_lls = re.search(r'l.+l',my_str)
print("Longest between the L's:", long_lls.group())

# find the shortest strings starting and ending with ll
long_lls = re.findall(r'l.+?l',my_str)
print("Shortest between the L's:", long_lls)

# find all occurances of Cind and the rest of the word it starts at the START of the string
cind = re.findall(r'(Cind[\w]*?)[\W]',my_str)
print("Cinds:",cind)

# extract the model number that Cinderella did NOT play on... hint: it must start with XS and end with B
didnot = re.findall(r'XS[\w]+?B',my_str)
print("Model nums:",didnot)

# extract all serial numbers... hint: they start and end with 1 or more uppercase letters with numbers between
allnums = re.findall(r'S[\w]+?X',my_str)
print("Serial nums:",allnums)

# IDs start with X or Y and end with _####
# find the names of the people who have IDs in this sentence
my_str2 = 'First id is Xsally_2137 and the second is Yomar_5389'
names = re.findall(r'([\w]+?)[_][\d]+?',my_str2)
print("Names:", names)



# NOTES
# (...) capture and place in a group
# (?:...) part of the match but NOT into a separate group
# (?<=...) do NOT capture, but must come before the following match
# (?=...) do NOT capture, but must come after the following match

# Find pattern that starts with 'the' and ends with a space but don't include
# the space and follows an _ but don't include the _. 
str1 = 'hello therefore hello_thereby; so'
find_there = re.search(r'(?<=[_])the[\w]*?(?=[;])',str1)
print(find_there.group())

# Find pattern that starts with X or Y, has 2 digits, and ends with B
# but don't include the B in the match!
str2 = 'Model X25A versus Model Y25B and Y1B'
find_by = re.findall(r'[XY][\d]{2}[A]?(?=[B])?',str2)
print(find_by)