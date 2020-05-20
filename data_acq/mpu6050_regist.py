import smbus
import time

# MPU6050 registers address
REG_PWR_MGMT_1      = 0x6B
REG_ACCEL_XOUT_H    = 0x3B
REG_ACCEL_YOUT_H    = 0x3D
REG_ACCEL_ZOUT_H    = 0x3F
REG_TEMP_OUT_H      = 0x41
REG_ACCEL_CONFIG    = 0x1C
REG_SMPLRT_DIV      = 0x19
REG_CONFIG          = 0x1A
REG_FIFO_EN         = 0x23
REG_USER_CTRL       = 0x6A
REG_FIFO_COUNT_H    = 0x72
REG_FIFO_COUNT_L    = 0x73
REG_FIFO            = 0x74
REG_WHO_AM_I        = 0x75
REG_INT_ENABLE      = 0x38

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
    def __init__(self, i2c_addr, g_range, sample_rate, verbose=False, accel_ms=1, temp_ms=1):
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

        # Open I2C bus
        try:
            self.bus = smbus.SMBus(1)
        except:
            print('Error occured when opening bus')

        # Store class attributes
        self.MPU6050_I2C_ADDR = i2c_addr
        self.verbose = verbose
        
        # Power management (ensure chip not in sleep mode, activate temperature measurement)
        self.bus.write_byte_data(self.MPU6050_I2C_ADDR, REG_PWR_MGMT_1, 0x01)
        
        # Accelerometer range configuration
        self.bus.write_byte_data(self.MPU6050_I2C_ADDR, REG_ACCEL_CONFIG, RANGE_VAL[g_range][0])
        self.ACCEL_DIV = float(RANGE_VAL[g_range][1])

        # Internal private register config
        self.bus.write_byte_data(self.MPU6050_I2C_ADDR, REG_CONFIG, 0x00)

        # Sample rate divisor
        if sample_rate in SMPLRT_VAL:
            self.bus.write_byte_data(self.MPU6050_I2C_ADDR, REG_SMPLRT_DIV, SMPLRT_VAL[sample_rate])
        else:
            # Custom range divisor in form 8kHz / (1 + SMPRT_DIV)
            if 0x00 < sample_rate < 0xFF:
                self.bus.write_byte_data(self.MPU6050_I2C_ADDR, REG_SMPLRT_DIV, sample_rate)
            else:
                raise ValueError('Sample rate input out of range')

        # Internal FIFO configuration
        self.bus.write_byte_data(self.MPU6050_I2C_ADDR, REG_USER_CTRL, 0x44)

        # Data to be inserted to FIFO
        # Accelerometer or temperature data, or both
        self.bus.write_byte_data(self.MPU6050_I2C_ADDR, REG_FIFO_EN, 0x08)

        # Set data ready interrupt register
        self.bus.write_byte_data(self.MPU6050_I2C_ADDR, REG_INT_ENABLE, 0x01)


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
        fc_h = self.bus.read_byte_data(self.MPU6050_I2C_ADDR, REG_FIFO_COUNT_H)
        fc_l = self.bus.read_byte_data(self.MPU6050_I2C_ADDR, REG_FIFO_COUNT_L)

        # Merge bytes
        fc = ((fc_h << 8) | fc_l)

        return fc


    def get_accel_data(self):
        # Store start time (in nanoseconds)
        time_start = time.clock_gettime_ns(time.CLOCK_REALTIME)
        time_prev  = time_start

        # FIFO overflow
        if self.fifo_count() >= 1024:
            # Reset FIFO
            self.bus.write_byte_data(self.MPU6050_I2C_ADDR, REG_USER_CTRL, 0x44)

        # Measure acceleration
        accel_x = self.read_raw_data(REG_ACCEL_XOUT_H) / self.ACCEL_DIV
        accel_y = self.read_raw_data(REG_ACCEL_YOUT_H) / self.ACCEL_DIV
        accel_z = self.read_raw_data(REG_ACCEL_ZOUT_H) / self.ACCEL_DIV
        # temp = self.read_raw_data(REG_TEMP_OUT_H) / -100

        # Measure time period
        time_delta = time.clock_gettime_ns(time.CLOCK_REALTIME) - time_prev
        time_prev = time.clock_gettime_ns(time.CLOCK_REALTIME)

        if self.verbose:
            print('addr: {} \t time: {}ms \t x: {}g \t y: {}g \t z: {}g'.format(
                    hex(self.MPU6050_I2C_ADDR), time_delta / 1000000, accel_x, accel_y, accel_z))
        
        return (time_delta, accel_x, accel_y, accel_z)

if __name__ == '__main__':
    mpu1 = MPU6050(i2c_addr=0x68, g_range='2g', sample_rate=1000, accel_ms=1, temp_ms=1)
    mpu2 = MPU6050(i2c_addr=0x69, g_range='2g', sample_rate=1000, accel_ms=1, temp_ms=1)
    
    while True:
        accel1 = mpu1.get_accel_data()
        accel2 = mpu2.get_accel_data()
        accel_delta = [(accel1[i] - accel2[i]) for i in range(len(accel1))]
        print(accel1)
        print(accel2)
