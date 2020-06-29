/* MPU6050 I2C library for Accelerometer data only
 * Copyright (c) 2020 Julien Grossholtz - https://openest.io
 * Modified by Rasis Syauqi Buldan (Jun 2020) 
 */

#ifndef _MPU6050_PI_I2C
#define _MPU6050_PI_I2C

#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>

/* Device I2C Address */
#define MPU6050_I2C_ADDR_1 0x68
#define MPU6050_I2C_ADDR_2 0x69

/* Register Address */
#define REG_ACCEL_ZOUT_H 0x3F
#define REG_ACCEL_ZOUT_L 0x40
#define REG_PWR_MGMT_1 0x6B
#define REG_ACCEL_CONFIG 0x1C
#define REG_SMPRT_DIV 0x19
#define REG_CONFIG 0x1A
#define REG_FIFO_EN 0x23
#define REG_USER_CTRL 0x6A
#define REG_FIFO_COUNT_L 0x72
#define REG_FIFO_COUNT_H 0x73
#define REG_FIFO 0x74
#define REG_WHO_AM_I 0x75

/* Constant */
/* Acceleration measurement range */
#define RANGE_ACCEL_DIV_2G 16384
#define RANGE_ACCEL_DIV_4G 8192
#define RANGE_ACCEL_DIV_8G 4096
#define RANGE_ACCEL_DIV_16G 2048

/* Sample Rate */
#define SAMPLE_RATE_50HZ_REGVAL 0x9F
#define SAMPLE_RATE_100HZ_REGVAL 0x4F
#define SAMPLE_RATE_200HZ_REGVAL 0x27
#define SAMPLE_RATE_400HZ_REGVAL 0x13
#define SAMPLE_RATE_800HZ_REGVAL 0x09
#define SAMPLE_RATE_1000HZ_REGVAL 0x07
#define SAMPLE_RATE_2000HZ_REGVAL 0x03
#define SAMPLE_RATE_4000HZ_REGVAL 0x01
#define SAMPLE_RATE_8000HZ_REGVAL 0x00

/* Struct for acceleration data point */
typedef struct {
    uint32_t timeDelta;
    float xAccel;
    float yAccel;
    float zAccel;
} MPUAccel;

/* Store timestamp */
uint32_t nanos, lastNanos, startNanos;

/* Store register reading bytes */
uint16_t fifo_len = 0;
char accel_x_h, accel_x_l;
char accel_y_h, accel_y_l;
char accel_z_h, accel_z_l;
char temp_h, temp_l;


/* Write into I2C device data from user space */
/* To do: proper implementation */
void i2c_write(__u8 reg_address, __u8 val);

/* Read from I2C device data from user space */
/* To do: proper implementation */
char i2c_read(uint8_t reg_address);

/* Merge two uint8_t data into uint16_t */
uint16_t merge_bytes( uint8_t LSB, uint8_t MSB);

/* Combine two register data (16 bit) to int16_t */
int16_t two_complement_to_int( uint8_t LSB, uint8_t MSB);

/* Setup MPU on specific address (register setup) */
void initializeMPU(uint8_t i2c_addr);

/* Get Acceleration data from FIFO registers */
MPUAccel getAccel(uint8_t i2c_addr);

/* Print acceleration data measured */
void printAccelData(MPUAccel m);

#endif

