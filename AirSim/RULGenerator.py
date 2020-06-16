import socket

'''
#Versi lama RULGenerator
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("localhost",9988))

while True:
    a = input("Send RUL (%) Value, to exit type x: ")
    if (a == 'x'):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("localhost",9988))
        s.sendall(a.encode('utf-8'))
        s.close()
        break
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("localhost",9988))
        s.sendall(a.encode('utf-8'))

#Versi lama RULreceiver
def cekRUL():
    conn, addr = s.accept()
    data = conn.recv(1024)
    data = data.decode('utf-8')
    print('data received: ' + data + '%')
    client.simPrintLogMessage("Remaining Useful Life: ",data + '%',2)
'''

sock = socket.socket()
sock.connect(("127.0.0.1", 12346))
#sock.listen(3)

while True:
    s = input("Send RUL (%) Value, to exit type exit: ")

    if s == 'exit':
        print("Quitting")
        break
    else:
        try:
            #sock = socket.socket()
            #sock.connect(("127.0.0.1", 12346))
            sock.send(s.encode())

        except Exception as e:
            print (e)
            sock.connect(("127.0.0.1", 12346))
#sock.shutdown(socket.SHUT_RDWR)
sock.close()