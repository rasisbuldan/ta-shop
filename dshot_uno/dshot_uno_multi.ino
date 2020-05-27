/* DShot protocol throttle for Arduino (Uno) */
/* Pin: D7 */

/* Library */
#include "Dshot.h"

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
    Serial.println("Initializing...");

    /* ESC Attach on D6 and D7 */
    esc1.attach(6);
    esc2.attach(7);

    /* Set initial throttle */
    esc1.setThrottle(48);  // Set to initial
    esc2.setThrottle(48);
    
    delay(5000);

    /* Increasing throttle for smooth transition to loop */
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