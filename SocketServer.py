import socket

s = socket.socket()

host = socket.gethostname()
print host
port = 1234

s.bind((host,port))
s.listen(5)

processedRequestCount = 0

while True:
	c,addr = s.accept()
	print 'Got a connection from', addr
	processedRequestCount += 1
	c.send('Thank you for connecting ' + str(processedRequestCount))
	c.close()