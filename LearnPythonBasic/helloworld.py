#this is single line comment

'''
this is multiple line comment
line 2
line 3
'''
# print('hello world \
#  a new line')

# print(r'this is a raw string \raw string')

# print(3/2)

# a = [1,2,3,4,5]
# b = [9,8,7,6,5]

# d = []

# for x,y in zip(a,b):
#     d.append(x+y)
# print(d)

# s = ['a','b','c','d']
# for i,v in enumerate(s,1):
#     print(i,v)

f = open("130.txt")
for line in f:
    print(line, end='')
