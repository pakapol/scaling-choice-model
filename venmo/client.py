import socket, sys

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('localhost', int(sys.argv[1])))
clientsocket.send(' '.join(sys.argv[2:]).encode())
