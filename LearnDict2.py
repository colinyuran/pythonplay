d = {}
d[42] = 'who are you'
d['age'] = 42
print d
cleanRet = d.clear()
print cleanRet
print d

x  = {'user':'ran','machines': ['pc1','pc2','pc3']}
y = x.copy()
print y
y['user'] = 'tan'
y['machines'].append('pc4')
print x
print y

from copy import deepcopy
d = {}
d['names'] = ['tom','jerry','cat']
c = d.copy()
dc = deepcopy(d)
d['names'].append('fish')
print c
print dc

t = dict.fromkeys(['name','phone','address'])
print t
t1 = dict.fromkeys(['name','phone','address'],'unknown')
print t1

t2 = {}
print t2.get('name')
print t2.get('name','unknown')
t2['name'] = 'eric'
print t2.get('name','N/A')

d1 = {'name':'ran','phone':'12221','address':'china'}
print d1
print d1.items()

it = d1.iteritems()
print it
print list(it)

print d1.keys()
print list(d1.iterkeys())

d2 = {'x':1,'y':2}
print d2.pop('x')
print d2

d3 = {'beea':12,'deaf':132,'eada':4893}
print d3
print d3.popitem()
print d3

d4 = {}
print d4.setdefault('x',123)
print d4
print d4.setdefault('x',256)
print d4


