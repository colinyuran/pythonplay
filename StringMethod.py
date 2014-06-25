contents = 'what a wonderful day! It\'s OK.'
print contents.find('day')
print contents.find('waaaa')

print contents.find('day',10,18)
print contents.find('day',10,25)

seq = ['1','2','3']
sep = '+'
print sep.join(seq)

dirs = 'usr','home','bin'
sep = '/'
print sep.join(dirs)

print contents.lower()
print contents.title()

import string
print string.capwords(contents)

print contents.replace('at','eeeee')

print '1+2+3+5'.split('+')

c = '     remove space in beginning and end    '
print c.strip()

c = '!!!*! remove others * ! aaa www ***!!!'
print c.strip('*!')

table = string.maketrans('cs','kz')
print len(table)
print table[97:123]

s = 'this is an incrediable test'
print s.translate(table)
print s.translate(table,' ')