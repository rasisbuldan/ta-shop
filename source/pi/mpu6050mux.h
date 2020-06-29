#ifndef _MPU6050_PI_I2C_MUX
#define _MPU6050_PI_I2C_MUX

#include "mpu6050.h"

#define TCA_ADDR 0x70

typedef struct {
    uint32_t timeDelta;
    MPUAccel mpu1;
    MPUAccel mpu2;
    MPUAccel mpu3;
    MPUAccel mpu4;
} MPUMux;

MPUMux mpuData;

/* Select MPU chip */
void selectMPU(uint8_t i);

#endif