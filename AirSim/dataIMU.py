import setup_path 
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2
import orbit
import time

import paho.mqtt.client as mqtt
import time
import random

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
#client.enableApiControl(True)
#client.armDisarm(True)

#client.takeoffAsync().join()
#client.hoverAsync().join()

def on_connect(Mclient, userdata, flags, rc):
     Mclient.connected_flag=True

def on_message(Mclient, userdata, message):
   print("Message Recieved: "+message.payload.decode())

def on_publish(mosq, obj, mid):
    print("Publish count: " + str(mid))


#MQTT AUTHENTICATION
Mclient = mqtt.Client()
Mclient.username_pw_set("strmgqzh", "UX6QWdM_-8Vm")
Mclient.connect('postman.cloudmqtt.com', 11876, 60)


Mclient.on_connect = on_connect
Mclient.on_message = on_message
Mclient.on_publish = on_publish

i =1

while True:
    with open(r'simHover.txt', 'a') as file:
        data = str(client.getImuData().time_stamp) + " " + str(client.getImuData().linear_acceleration.x_val) + " " + str(client.getImuData().linear_acceleration.y_val) + " " + str(client.getImuData().linear_acceleration.z_val)
        #file.write(','.join(data.split(" ")) + '\n')
    i = i+1
    time.sleep(0.001)
    if i % 1500 == 0:
        Mclient.publish("/flow", client.getImuData().linear_acceleration.x_val)
        Mclient.publish("/pressure", client.getImuData().linear_acceleration.y_val)
        Mclient.publish("/temperature", client.getImuData().linear_acceleration.z_val)
        Mclient.publish("/power", random.uniform(0.48, 0.58))
        Mclient.publish("/hours", 10)


client.armDisarm(False)
client.reset()
client.enableApiControl(False)