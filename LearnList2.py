chars = list('hello')
print chars
print ''.join(chars)
print '.'.join(chars)

numbers = [1,1,1]
numbers[1] = 2
print numbers

# this is invalid
# numbers[10] = 100
# print numbers

names = ['ran','cat','dog','fish','ball']
print names
del names[2]
print names
print len(names)

name = list('Perl')
print name
name[2:] = list('ar')
print name

name = list('Perl')
print name
name[1:] = list('ython')
print name

numbers = [1,5]
print numbers
numbers[1:1] = [2,3,4]
print numbers
numbers[2:4] = []
print numbers

