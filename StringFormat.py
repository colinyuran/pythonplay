from string import Template
s = Template('$x, glorious $x!')
print s.substitute(x='slurm')

s = Template('It\'s ${x}tastic!')
print s.substitute(x='slurm')

s = Template('Make $$ selling $x!')
print s.substitute(x='slurm')

s = Template('A $thing must never $action')
d = {}
d['thing'] = 'gentleman'
d['action'] = 'show his socks'
print s.substitute(d)

print '%s plus %s equals %s' % (1,1,2)

print 'Price of eggs: $%d' % 42
print 'Price of eggs in hex: %x' % 42
from math import pi
print 'Pi: %f' % pi
print 'Short Pi: %i' % pi
print 'Using str: %s' % 42L
print 'Using repr: %r' % 42L

print
print '%10f' % pi
print '%10.2f' % pi
print '%.2f' % pi

print '%.5s' % 'Whose is your'
print '%.*s' % (3,'whoaw is ok')
print '%010.2f' % pi

print '%-10.2f' % pi
print '% 5d' % 10 
print '% 5d' % -10
print '%+5d' % 10 
print '%+5d' % -10