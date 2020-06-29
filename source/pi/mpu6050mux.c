#include "mpu6050mux.h"

/* Select MPU chip */
void selectMPU(uint8_t i) {

}

/* Get all acceleration data from MPU6050 (currently hardcoded with 4 devices) */
void getAllMPU(MPUMux *mmux) {
    (*mmux).mpu1 = getAccel(TCA_ADDR + 0x00);
    (*mmux).mpu2 = getAccel(TCA_ADDR + 0x01);
    (*mmux).mpu3 = getAccel(TCA_ADDR + 0x02);
    (*mmux).mpu4 = getAccel(TCA_ADDR + 0x03);
    (*mmux).timeDelta = ((*mmux).mpu1.timeDelta + (*mmux).mpu2.timeDelta + (*mmux).mpu3.timeDelta + (*mmux).mpu4.timeDelta) / 4;
}

void printMPUData()