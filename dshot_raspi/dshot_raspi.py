import RPi.GPIO as GPIO
import time
import gc

# Hopefully reduce interruptions
gc.disable()

ESC_PIN = 5

# DSHOT150 TIMINGS (1.67us per frame)
T0H = 0.000000625  # T0H = 2500ns
T0L = 0.000001045  # T0L = 4180ns
T1H = 0.000001250  # T1H = 5000ns
T1L = 0.000000420  # T1L = 1680ns

INTER_PACKET_DELAY = 0.00002  # 20us

def transmit_one():
    GPIO.output(ESC_PIN, GPIO.HIGH)
    time.sleep(T1H)
    GPIO.output(ESC_PIN, GPIO.LOW)
    time.sleep(T1L)

def transmit_zero():
    GPIO.output(ESC_PIN, GPIO.HIGH)
    time.sleep(T0H)
    GPIO.output(ESC_PIN, GPIO.LOW)
    time.sleep(T0L)

def inter_packet_delay():
    time.sleep(INTER_PACKET_DELAY)

def transmit_value(value, telemetry=0):
    # Add telemetry bit
    packet_data = (value << 1) | (1 if telemetry else 0) # Calculate and add CRC 
    crc_sum = (packet_data ^ (packet_data >> 4) ^ (packet_data >> 8)) & 0xf
    packet_data = (packet_data << 4) | crc_sum # Output the packet
    for bit in range(16): 
        if (packet_data >> (15 - bit)) & 1:
            transmit_one()
        else:
            transmit_zero()

if __name__ == "__main__":
    # Setup GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ESC_PIN, GPIO.OUT)

    # Send reset value to arm the ESC
    print("RESETTING")
    for _ in range(20):
        transmit_value(0)
        inter_packet_delay()
    time.sleep(5)

    # Send values from 48 (throttle = 0) to 100
    for throttle in range(400,450, 1):
        print("Throttle = {}".format(throttle))
        for _ in range(1000):
            transmit_value(throttle)
            inter_packet_delay()
    
    

    # Send reset again
    print("RESETTING")
    for _ in range(10):
        transmit_value(100)
        inter_packet_delay()