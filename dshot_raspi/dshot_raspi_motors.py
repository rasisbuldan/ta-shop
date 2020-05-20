# From github.com/dmrlawson/raspberrypi-dshot
# Dshot-150 timing
# Throttle: 48(0) - 2048(2000)

ESC_PIN = 5

import dshot
import time

for _ in range(10000):
    dshot.send(48, ESC_PIN)

for _ in range(20000):
    dshot.send(99, ESC_PIN)

for _ in range(10000):
    dshot.send(199, ESC_PIN)