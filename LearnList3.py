lst = [3,4,5]
lst.append(7)
print lst

numbers = [1,5,6,8,6,9,6,0,5]
print numbers.count(5)
print numbers.count(6)

embeded = [[1,2],1,2,3,2,[[1,2],1]]
print embeded.count([1,2])

a = [1,2,3]
b = [4,5,6]
a.extend(b)
print a

knights = ['we','are','the','knights','who','say','ni']
print knights.index('who')
# will throw exception
# print knights.index('whole')

a = [1,2,3,4,5,6]
a.insert(2,'two')
print a

print a.pop()
print a

print a.pop(2)
print a

print a.pop(0)
print a

x = ['to','be','or','not','to','be']
x.remove('be')
print x

x = [1,2,3]
x.reverse()
print x

x = [0,5,-1,2,9]
x.sort()
print x

x = ['cat','dog','apple','fruit','a','where']
x.sort(key=len)
print x

x = ['cat','dog','apple','fruit','a','where']
x.sort(reverse=True)
print x
