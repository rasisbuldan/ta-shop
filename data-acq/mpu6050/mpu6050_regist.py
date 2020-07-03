import smbus
import time
import numpy as np

# INA260 Voltage-Current Sensor
import board
import adafruit_ina260

# Measurement range dictionary (register value, MSB count)
RANGE_VAL = {
    '2g'  : (0x00, 16384),
    '4g'  : (0x01, 8192),
    '8g'  : (0x02, 4096),
    '16g' : (0x03, 2048)
}

# Sample rate dictionary
SMPLRT_VAL = {
    50   : 0x9F,
    100  : 0x4F,
    200  : 0x27,
    250  : 0x1F,
    400  : 0x13,
    800  : 0x09,
    1000 : 0x07,
    2000 : 0x03,
    4000 : 0x01,
    8000 : 0x00
}

class MPU6050:
    # MPU6050 registers address
    __REG_PWR_MGMT_1      = 0x6B
    __REG_ACCEL_XOUT_H    = 0x3B
    __REG_ACCEL_YOUT_H    = 0x3D
    __REG_ACCEL_ZOUT_H    = 0x3F
    __REG_TEMP_OUT_H      = 0x41
    __REG_ACCEL_CONFIG    = 0x1C
    __REG_SMPLRT_DIV      = 0x19
    __REG_CONFIG          = 0x1A
    __REG_FIFO_EN         = 0x23
    __REG_USER_CTRL       = 0x6A
    __REG_FIFO_COUNT_H    = 0x72
    __REG_FIFO_COUNT_L    = 0x73
    __REG_FIFO            = 0x74
    __REG_WHO_AM_I        = 0x75
    __REG_INT_ENABLE      = 0x38

    def __init__(self, i2c_addr, g_range, sample_rate, verbose=False, accel_ms=1, temp_ms=1, include_ina=False):
        '''
        Initialization
        args:
        - i2c_addr       : i2c address of device (default: 0x68)
        - g_range        : specify measurement range as listed in RANGE_VAL
        - sample_rate    : specify sample rate as listed in SMPLRT_VAL
        - verbose        : verbose output
        - accel_ms (0/1) : acceleration measurement
        - temp_ms (0/1)  : temperature measurement
        '''
        self.x_offset = 0
        self.y_offset = 0
        self.z_offset = 0

        # Open I2C bus
        try:
            self.bus = smbus.SMBus(1)
        except:
            print('Error occured when opening bus')

        # Store class attributes
        self.MPU6050_I2C_ADDR = i2c_addr
        self.verbose = verbose
        
        # Power management (ensure chip not in sleep mode, activate temperature measurement)
        self.bus.write_byte_data(self.MPU6050_I2C_ADDR, self.__REG_PWR_MGMT_1, 0x01)
        
        # Accelerometer range configuration
        self.bus.write_byte_data(self.MPU6050_I2C_ADDR, self.__REG_ACCEL_CONFIG, RANGE_VAL[g_range][0])
        self.ACCEL_DIV = float(RANGE_VAL[g_range][1])

        # Internal private register config
        self.bus.write_byte_data(self.MPU6050_I2C_ADDR, self.__REG_CONFIG, 0x00)

        # Sample rate divisor
        if sample_rate in SMPLRT_VAL:
            self.bus.write_byte_data(self.MPU6050_I2C_ADDR, self.__REG_SMPLRT_DIV, SMPLRT_VAL[sample_rate])
        else:
            # Custom range divisor in form 8kHz / (1 + SMPRT_DIV)
            if 0x00 < sample_rate < 0xFF:
                self.bus.write_byte_data(self.MPU6050_I2C_ADDR, self.__REG_SMPLRT_DIV, sample_rate)
            else:
                raise ValueError('Sample rate input out of range')

        # Internal FIFO configuration
        self.bus.write_byte_data(self.MPU6050_I2C_ADDR, self.__REG_USER_CTRL, 0x44)

        # Data to be inserted to FIFO
        # Accelerometer or temperature data, or both
        self.bus.write_byte_data(self.MPU6050_I2C_ADDR, self.__REG_FIFO_EN, 0x08)

        # Set data ready interrupt register
        self.bus.write_byte_data(self.MPU6050_I2C_ADDR, self.__REG_INT_ENABLE, 0x01)

        if include_ina:
            # INA260 object declaration
            i2c = board.I2C()
            ina260 = adafruit_ina260.INA260(i2c)
            ina260.averaging_count = adafruit_ina260.AveragingCount.COUNT_1
            ina260.current_conversion_time = adafruit_ina260.ConversionTime.TIME_140_us
            ina260.voltage_conversion_time = adafruit_ina260.ConversionTime.TIME_140_us


    def read_raw_data(self, reg_addr):
        # 16-bit data (accelerometer and gyro)
        val_h = self.bus.read_byte_data(self.MPU6050_I2C_ADDR, reg_addr)
        val_l = self.bus.read_byte_data(self.MPU6050_I2C_ADDR, reg_addr + 0x01)

        val = ((val_h << 8) | val_l)

        # Convert to signed value
        if val > 32768:
            val = val - 65536

        return val


    def fifo_count(self):
        fc_h = self.bus.read_byte_data(self.MPU6050_I2C_ADDR, self.__REG_FIFO_COUNT_H)
        fc_l = self.bus.read_byte_data(self.MPU6050_I2C_ADDR, self.__REG_FIFO_COUNT_L)

        # Merge bytes
        fc = ((fc_h << 8) | fc_l)

        return fc


    def reset_offset(self):
        # Averaging data from 100 accel samples
        avg_array = [[],[],[]]
        for i in range(100):
            accel = self.get_accel_data()
            avg_array[0].append(accel[1])
            avg_array[1].append(accel[2])
            avg_array[2].append(accel[3])

        self.x_offset = np.average(avg_array[0])
        self.y_offset = np.average(avg_array[1])
        self.z_offset = np.average(avg_array[2])


    def get_accel_data(self):
        # Store start time (in nanoseconds)
        time_start = time.clock_gettime_ns(time.CLOCK_REALTIME)
        time_prev  = time_start

        # FIFO overflow
        if self.fifo_count() >= 1024:
            # Reset FIFO
            self.bus.write_byte_data(self.MPU6050_I2C_ADDR, self.__REG_USER_CTRL, 0x44)

        # Measure acceleration
        accel_x = (self.read_raw_data(self.__REG_ACCEL_XOUT_H) / self.ACCEL_DIV) - self.x_offset
        accel_y = (self.read_raw_data(self.__REG_ACCEL_YOUT_H) / self.ACCEL_DIV) - self.y_offset
        accel_z = (self.read_raw_data(self.__REG_ACCEL_ZOUT_H) / self.ACCEL_DIV) - self.z_offset
        
        # INA260
        voltage = self.ina260.voltage
        current = self.ina260.current

        # Measure time period
        time_delta = time.clock_gettime_ns(time.CLOCK_REALTIME) - time_prev
        time_prev = time.clock_gettime_ns(time.CLOCK_REALTIME)

        if self.verbose:
            print('addr: {} \t time: {}ms \t x: {}g \t y: {}g \t z: {}g'.format(
                    hex(self.MPU6050_I2C_ADDR), time_delta / 1000000, accel_x, accel_y, accel_z))
        
        return (time_delta, accel_x, accel_y, accel_z, voltage, current)

    ### Barbaric ###
    def get_combined_data(self):


if __name__ == '__main__':
    mpu1 = MPU6050(i2c_addr=0x68, g_range='4g', sample_rate=1000)
    #mpu2 = MPU6050(i2c_addr=0x69, g_range='2g', sample_rate=1000)
    mpu1.reset_offset()
    #mpu2.reset_offset()

    while True:
        accel1 = mpu1.get_accel_data()
        #accel2 = mpu2.get_accel_data()
        #accel_delta = [(accel1[i] - accel2[i]) for i in range(len(accel1))]
        print(accel1)
        #print(accel2)
