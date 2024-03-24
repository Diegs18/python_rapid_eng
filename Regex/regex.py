
import re
str = '### Subject: scuba diving at… From: nick.digs@asu.edu Body: stUnderwateris, where it '

match = re.search(r':.*:',str)
print("1:",match.group()) #: scuba diving at… From: steve.millman@asu.edu Body:

match = re.search(r'st\S*',str)
print("2:",match.group()) #steve.millman@asu.edu

match = re.search(r':.*?:',str)
print("3:",match.group()) #: scuba diving at… From:

match = re.search(r'[\w ]*is$',str)
print("4:", match.group()) # where it is at


match = re.search(r'[\w]+[.]?[\w]*?@[\w]+[.][\w]+',str)
print("Email:", match.group()) #steve.millman


