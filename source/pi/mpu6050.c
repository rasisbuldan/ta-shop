/* MPU6050 I2C library for Accelerometer data only
 * Copyright (c) 2020 Julien Grossholtz - https://openest.io
 * Modified by Rasis Syauqi Buldan (Jun 2020) 
 */

#include "mpu6050.h"

// Please note, this is not the recommanded way to write data
// to i2c devices from user space.
void i2c_write(__u8 reg_address, __u8 val) {
	char buf[2];
	if(file < 0) {
		printf("Error, i2c bus is not available\n");
		exit(1);
	}

	buf[0] = reg_address;
	buf[1] = val;

	if (write(file, buf, 2) != 2) {
		printf("Error, unable to write to i2c device\n");
		exit(1);
	}

}


// Please note, this is not thre recommanded way to read data
// from i2c devices from user space.
char i2c_read(uint8_t reg_address) {
	char buf[1];
	if(file < 0) {
		printf("Error, i2c bus is not available\n");
		exit(1);
	}

	buf[0] = reg_address;

	if (write(file, buf, 1) != 1) {
		printf("Error, unable to write to i2c device\n");
		exit(1);
	}


	if (read(file, buf, 1) != 1) {
		printf("Error, unable to read from i2c device\n");
		exit(1);
	}

	return buf[0];

}


uint16_t merge_bytes( uint8_t LSB, uint8_t MSB) {
	return  (uint16_t) ((( LSB & 0xFF) << 8) | MSB);
}


// 16 bits data on the MPU6050 are in two registers,
// encoded in two complement. So we convert those to int16_t
int16_t two_complement_to_int( uint8_t LSB, uint8_t MSB) {
	int16_t signed_int = 0;
	uint16_t word;

	word = merge_bytes(LSB, MSB);

	if((word & 0x8000) == 0x8000) { // negative number
		signed_int = (int16_t) -(~word);
	} else {
		signed_int = (int16_t) (word & 0x7fff);
	}

	return signed_int;
}


/* Setup MPU on specific address (register setup) */
void initializeMPU(uint8_t i2c_addr){
    int adapter_nr = 1; /* probably dynamically determined */
    char bus_filename[250];

    snprintf(bus_filename, 250, "/dev/i2c-1", adapter_nr);
	file = open(bus_filename, O_RDWR);
	if (file < 0) {
		/* ERROR HANDLING; you can check errno to see what went wrong */
		exit(1);
	}


	if (ioctl(file, I2C_SLAVE, MPU6050_I2C_ADDR) < 0) {
		/* ERROR HANDLING; you can check errno to see what went wrong */
		exit(1);
	}

    i2c_write(REG_PWR_MGMT_1, 0x01);
	i2c_write(REG_ACCEL_CONFIG, 0x01); // 4G
	i2c_write(REG_SMPRT_DIV, SAMPLE_RATE_1000HZ_REGVAL); // 1000Hz
	i2c_write(REG_CONFIG, 0x00);
	i2c_write(REG_FIFO_EN, 0x88);
	i2c_write(REG_USER_CTRL, 0x44);

    lastNanos = 0;
}


/* Get Acceleration data from FIFO registers */
MPUAccel getAccel(uint8_t i2c_addr){
	MPUAccel m;

    accel_x_h = i2c_read(REG_FIFO_COUNT_L);
    accel_x_l = i2c_read(REG_FIFO_COUNT_H);
    fifo_len = merge_bytes(accel_x_h,accel_x_l);

    /* FIFO overflow */
    if (fifo_len >= 1024) {
        printf("fifo overflow! \n");
        i2c_write(REG_USER_CTRL, 0x44);
        continue;
    }

    if (fifo_len >= 8) {
        accel_x_h = i2c_read(REG_FIFO);
        accel_x_l = i2c_read(REG_FIFO);
        accel_y_h = i2c_read(REG_FIFO);
        accel_y_l = i2c_read(REG_FIFO);
        accel_z_h = i2c_read(REG_FIFO);
        accel_z_l = i2c_read(REG_FIFO);
    }
    
    m.xAccel = ((float) two_complement_to_int(accel_x_h, accel_x_l) / RANGE_ACCEL_DIV_4G);
    m.yAccel = ((float) two_complement_to_int(accel_y_h, accel_y_l) / RANGE_ACCEL_DIV_4G);
    m.zAccel = ((float) two_complement_to_int(accel_z_h, accel_z_l) / RANGE_ACCEL_DIV_4G);
    m.timeDelta = (nanos - lastNanos);
    lastNanos = nanos;

	return m;
}

/* Print acceleration data measured */
void printAccelData(MPUAccel m){
	printf("Td: %ldns \t x: %.5f \t y: %.5f \t z: %.5f\n", m.timeDelta, m.xAccel, m.yAccel, m.zAccel);
}