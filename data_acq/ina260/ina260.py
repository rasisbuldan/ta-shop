'''
TI INA260 Current Logging (I2C-register-based)
Datasheet:  http://www.ti.com/lit/ds/symlink/ina260.pdf?ts=1590430404379
I2C address: 0x44
'''

import smbus
import time

class INA260:
    # INA260 registers address
    __REG_CONFIG            = 0x00
    __REG_CURRENT           = 0x01
    __REG_BUS_VOLTAGE_ADDR  = 0x02

    # Divider value
    BUS_VOLTAGE_LSB = 1.25
    CURRENT_LSB     = 1.25

    def __init__(self, i2c_addr, verbose=False):
        '''
        Initialization
        '''
        
        # Open I2C bus
        try:
            self.bus = smbus.SMBus(1)
        except:
            print('Error occured when opening bus')
        
        self.INA260_I2C_ADDR = i2c_addr
        self.verbose = verbose


    def twos_compliment_to_int(self, val, len):
        '''
        Convert two's complement to integer with len bit
        '''
        if(val & (1 << len - 1)):
			val = val - (1 << len)
        
        return val


    def reset(self):
        '''
        Reset value stored in register
        '''
        self.bus.write_i2c_block_data(self.INA260_I2C_ADDR, self.__REG_CONFIG, [0x80, 0x00])


    def get_bus_voltage(self):
        '''
        Get bus voltage
        '''
        
        # Read 2 blocks of data
        raw_vbus = self.bus.read_i2c_block_data(self.INA260_I2C_ADDR, self.__REG_BUS_VOLTAGE_ADDR, 2)
        data_vbus = raw_vbus[0] * 256 + raw_vbus[1]
        val_vbus = float(data_vbus) / 1000.0 * self.BUS_VOLTAGE_LSB

        return val_vbus

    
    def get_current(self):
        raw_current = self.bus.read_i2c_block_data(self.INA260_I2C_ADDR, self.__REG_CURRENT, 2)
        data_current = raw_current[0] * 256 + raw_current[1]
        sign_current = data_current >> 15

        # Signed (negative)
        if sign_current:
            val_current = float(self.twos_compliment_to_int(data_current, 16)) / 1000.0 * self.CURRENT_LSB
        
        else:
            val_current = float(data_current) / 1000.0 * self.CURRENT_LSB

if __name__ == '__main__':
    ina = INA260(i2c_addr=0x44, verbose=True)