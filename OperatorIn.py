permissions = 'rw'
print 'w' in permissions
print 'x' in permissions

users = ['ran','toa','olw']
name = raw_input('Enter your name:')
print name in users

subject = '$$$ who are you $$$'
print '$$$' in subject

database = [
	['ran','1234'],
	['col','2345'],
	['cat','cat123']
]

userName = raw_input('User Name:')
pin = raw_input('Pin:')
if [userName,pin] in database:
	print 'Access granted'
else:
	print 'Access denied'