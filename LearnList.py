# coding=utf-8

numbers = [1,2,3,4,5,6,7,8,9,10]

print 'Number at 0: ' + str(numbers[0])
print 'Number at 9: ' + str(numbers[9])
print 'Number at -1: ' + str(numbers[-1])
print 'Number at -2: ' + str(numbers[-2])

# 分片包含三个参数，第一个是起始下标，第二个是终止下标，第三个是步长
# 步长是正数时，起始下标对应的元素在整个序列中必须出现在终止下标对应的元素的左侧
# 输出的分片序列中包含起始下标对应的元素，但不包含终止下标对应的元素
# 步长是负数时，起始下标对应的元素在整个序列中必须出现在终止下标对应的元素的右侧
# 输出的分片序列中包含起始下标对应的元素，但不包含终止下标对应的元素

print 'numbers[3:6]'
print numbers[3:6]

print 'numbers[7:10]'
print numbers[7:10]

print 'numbers[-3:-1]'
print numbers[-3:-1]

print 'numbers[-3:0]'
print numbers[-3:0]

print 'numbers[-6:-3]'
print numbers[-6:-3]

print 'numbers[-3:]'
print numbers[-3:]

print 'numbers[:3]'
print numbers[:3]

print 'numbers[:]'
print numbers[:]

print 'numbers[:-3]'
print numbers[:-3]

print 'numbers[0:10:3]'
print numbers[0:10:3]

print 'numbers[0:9:3]'
print numbers[0:9:3]

print 'numbers[10:0:-2]'
print numbers[10:0:-2]

print 'numbers[10:1:-2]'
print numbers[10:1:-2]

print 'numbers[11:1:-2]'
print numbers[11:1:-2]

print 'numbers[9:0:-2]'
print numbers[9:0:-2]

print 'numbers[9:1:-2]'
print numbers[9:1:-2]

print 'numbers[8:1:-2]'
print numbers[8:1:-2]

print 'numbers[8:2:-2]'
print numbers[8:2:-2]

print 'numbers[-1:2:-2]'
print numbers[-1:2:-2]

print 'numbers[-2:2:-2]'
print numbers[-2:2:-2]


addNumbers = [3,4,5] + [10,11,12]
print addNumbers

multipyNumbers = [3,4] * 10
print multipyNumbers

empty = []
print empty

emptyWith10None = [None] * 10
print emptyWith10None

print len(numbers)
print max(numbers)
print min(numbers)
print max(2,5,9,1)
print min(0,-1,19,20)


