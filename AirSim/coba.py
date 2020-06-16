import setup_path 
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2
import orbit

import socket
import threading

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.simPrintLogMessage("Time to maintenance (hour): ","500",2)

def cekRUL():
    sock = socket.socket()
    sock.bind(("127.0.0.1",12346))
    sock.listen(3)
    conn = sock.accept()

    while True: 
        try:
            m = conn[0].recv(4096)
            RUL = m.decode()
            client.simPrintLogMessage("Time to maintenance (hour): ",RUL + '%',2)
        except:
            continue

    sock.shutdown(socket.SHUT_RDWR)
    sock.close()

thread = threading.Thread(target=cekRUL)
thread.daemon= True
thread.start()
while True:
    a = input("w for take off, s for landing, d & a to orbit, space to hover, x to exit: ")
    landed = client.getMultirotorState().landed_state

    if a == 'w': #Drone Take Off
        if landed == airsim.LandedState.Landed:
            print("taking off...")
            client.takeoffAsync().join()
        else:
            print("already flying...")
            client.hoverAsync().join()
        
       
    
    elif a == 's': #Drone Landing
       
        if landed == airsim.LandedState.Landed:
            print("already landed...")
        else:
            print("landing...")
            client.landAsync().join()
        

    elif a == ' ':
        
        client.hoverAsync().join()
        
    
    elif a == 'd': #orbit
       
        nav = orbit.OrbitNavigator(10 ,20, 3, 1, [1,0], 0)
        nav.start()
        
    
    elif a == 'a': #orbit
        
        nav = orbit.OrbitNavigator(10 ,20, 3, 1, [1,0], 0)
        nav.start()
        

    elif a == 'x': #exit
        
        client.armDisarm(False)
        client.reset()
        client.enableApiControl(False)
        break
   

    



'''
while True:
    conn, addr = s.accept()
    data = conn.recv(1024)
    data = data.decode('utf-8')
    print('data received: ' + data + '%')
    client.simPrintLogMessage("Remaining Useful Life: ",data + '%',2)
    if data == 'x':
        conn.close()
        break
    else:
        continue
'''
