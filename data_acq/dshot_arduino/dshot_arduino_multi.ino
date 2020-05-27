/* Bit-banging method
 * DShot600 for 16MHz CPU (Arduino Uno)
 *      T0H: 10 cycles
 *      T0L: 17 cycles
 *      T1H: 20 cycles
 *      T1L: 7  cycles
 * ---------------------------
 *           27 cycles per bit
 */

/* DShot protocol throttle for Arduino (Uno) */
/* Pin: D7 */

/* Library */
#include "DShot.h"

/* Object declaration */
DShot esc1;
DShot esc2;

/* Global Variable */
uint16_t throttle = 0;
uint16_t target = 0;
uint16_t min_throttle = 200;
uint16_t max_throttle = 400;
uint16_t const_throttle = 100;

void setup() {
    Serial.begin(115200);

    /* ESC Attach (D7 on Arduino UNO) */
    Serial.println("Initializing...");
    esc1.attach(6);
    esc2.attach(7);
    esc1.setThrottle(48);  // Set to initial
    esc2.setThrottle(48);  
    delay(5000);

    /* for (throttle = 48; throttle <= min_throttle; throttle++){
        esc1.setThrottle(throttle);
        esc2.setThrottle(throttle);
    } */
}

void loop() {
    for (throttle = min_throttle; throttle <= max_throttle; throttle++) {
        Serial.print("Throttle: ");
        Serial.print(throttle);
        Serial.print(" | ");
        Serial.println(min_throttle + (max_throttle - throttle));
        esc1.setThrottle(throttle);
        esc2.setThrottle(min_throttle + (max_throttle - throttle));
        delay(50);
    }
    delay(500);

    for (throttle = max_throttle; throttle >= min_throttle; throttle--) {
        Serial.print("Throttle: ");
        Serial.print(throttle);
        Serial.print(" | ");
        Serial.println(min_throttle + (max_throttle - throttle));
        esc1.setThrottle(throttle);
        esc2.setThrottle(min_throttle + (max_throttle - throttle));
        delay(50);
    }
    delay(500);
}