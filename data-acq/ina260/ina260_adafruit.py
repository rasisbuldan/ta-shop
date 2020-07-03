import time
import board
import adafruit_ina260

i2c = board.I2C()

ina260 = adafruit_ina260.INA260(i2c)
ina260.averaging_count = adafruit_ina260.AveragingCount.COUNT_256

time0 = time.perf_counter_ns()
while True:
    #print(ina260.averaging_count)
    print("Current: {:.2f}mA | Voltage: {:.2f}V | Power:  {:.2f}mW | Time: {:.2f}ms".format(ina260.current, ina260.voltage, ina260.power, (time.perf_counter_ns() - time0)/1000000), end='\n')
    time0 = time.perf_counter_ns()
