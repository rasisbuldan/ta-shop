/* DShot protocol throttle for Arduino (Uno) */
/* Pin: D7 */

/* Library */
#include <DShot.h>

/* Object declaration */
DShot esc1;

/* Global Variable */
uint16_t throttle = 0;
uint16_t target = 0;
uint16_t min_throttle = 800;
uint16_t max_throttle = 900;
uint16_t const_throttle = 400;

void setup() {
    Serial.begin(115200);

    /* ESC Attach (D7 on Arduino UNO) */
    Serial.println("Initializing...");
    esc1.attach(7);
    esc1.setThrottle(throttle);  // Set to initial
}

void loop() {
    Serial.println("Loop..");
    esc1.setThrottle(const_throttle);
    delay(20);

    /* for (throttle = min_throttle; throttle <= max_throttle; throttle++) {
        Serial.print("Throttle: ");
        Serial.println(throttle);
        esc1.setThrottle(throttle);
        delay(50);
    }
    delay(500);

    for (throttle = max_throttle; throttle >= min_throttle; throttle--) {
        Serial.print("Throttle: ");
        Serial.println(throttle);
        esc1.setThrottle(throttle);
        delay(50);
    }
    delay(500); */
}