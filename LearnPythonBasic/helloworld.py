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

# def fib(n):
#     """
#     this is a function to calculate fib
#     """

#     result = [0,1]
#     for i in range(n-2):
#         result.append(result[-1] + result[-2])
#     return result

# lst = fib(10)
# print(lst)

# print(fib.__doc__)

# def foo(x,y,z, *args, **kargs):
#     print(x)
#     print(y)
#     print(z)
#     print(args)
#     print(kargs)

# foo(1,2,3)
# foo(1,2,3,4,5,6)
# foo(1,2,3,4,5,6,name='who', key='why')

def foo(fun):
    def wrap():
        print("start")
        fun()
        print("end")
        print(fun.__name__)
    return wrap

@foo
def bar():
    print("I am in bar()")

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
