phoneBook = {'cat':'2341','dog':'0123','duck':'2334'}
print phoneBook

print phoneBook['dog']

items = [('name','guuy'), ('age',23)]
d = dict(items)
print d

d = dict(name='cat',age=42,weight=100)
print d

x = {}
x[42] = 'foo'
print x[42]

people = {
	'Alice': {
		'phone': '1234',
		'age': 42
	},

	'Beth' : {
		'phone':'25323',
		'age':20
	},
	'Cecil' : {
		'phone': '00011',
		'age': 100
	}
}

labels = {
	'phone' : 'Phone Number',
	'age' : 'age'
}

name = raw_input('Name:')

request = raw_input('phone number(p) or age(a)?')

if request == 'p' : key ='phone'
if request == 'a' : key = 'age'

if name in people:
	print "%s's %s is %s." % (name,labels[key],people[name][key])
	