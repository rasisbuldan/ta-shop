/* DShot protocol throttle for Arduino (Uno) */
/* Pin: D7 */

/* Library */
#include "Dshot.h"

/* Object declaration */
DShot esc1;

/* Global Variable */
uint16_t throttle = 0;
uint16_t target = 0;
uint16_t min_throttle = 200;
uint16_t max_throttle = 550;
uint16_t const_throttle = 500;

void setup() {
    Serial.begin(115200);

    /* ESC Attach (D7 on Arduino UNO) */
    Serial.println("Initializing...");
    esc1.attach(9);
    
    esc1.setThrottle(48);  // Set to initial
    delay(5000);
}

void loop() {
    Serial.println("Loop..");
    esc1.setThrottle(const_throttle);
    delay(50);
}
