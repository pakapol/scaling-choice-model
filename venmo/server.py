#!/usr/bin/python
import socket, sys 

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('localhost', int(sys.argv[1])))
serversocket.listen(5) # become a server socket, maximum 5 connections

while True:
    connection, address = serversocket.accept()
    buf = connection.recv(2048)
    if len(buf) > 0:
        sys.stdout.write(buf.decode() + '\n')
        sys.stdout.flush()
