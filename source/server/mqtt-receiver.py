import paho.mqtt.client as mqtt
import time
import json
import sys

mpudata = ""

def on_connect(client, userdata, flags, rc):
    print("Connection: ", rc)
    client.subscribe('topic/mpu6050')

def print_mpudata_json(payload):
    mpudata = json.loads(str(payload))
    print(json.dumps(mpudata, indent=2))

def on_message(client, userdata, msg):
    #print('Rcv message on', msg.topic, ':', str(msg.payload.decode("utf-8")))
    print_mpudata_json(msg.payload.decode("utf-8"))

# Initialize MQTT connection
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.enable_logger() #logger=None

client.connect('192.168.0.118', port=1884, keepalive=1800)

client.loop_forever()