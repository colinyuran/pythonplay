# coding=utf-8

format = 'Hello %s, %s is enough for you'
values = 'world','Hot'
print format % values

format = 'Pi with three decimal: %.3f'
from math import pi
print format % pi
