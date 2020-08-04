/* DShot protocol throttle for Arduino (Uno) */
/* Pin: D7 */

/* Library */
#include "Dshot.h"

/* Object declaration */
DShot esc1;

/* Global Variable */
long throttle = 0;
long target = 0;
long min_throttle = 200;
long max_throttle = 600;
long const_throttle = 500;
long random_val;

void setup() {
    Serial.begin(115200);

    /* ESC Attach (D7 on Arduino UNO) */
    Serial.println("Initializing...");
    esc1.attach(9);
    
    esc1.setThrottle(48);  // Set to initial
    delay(5000);
}

void loop() {
    random_val = random(3,7) * 100;
    
    /* Ascend to setpoint */
    for (long i = 1; i < 5; i++) {
        throttle = i * (random_val / 5);
        esc1.setThrottle(throttle);
        Serial.println(throttle);
        delay(5);
    }

    /* Add noise to setpoint for 10s */
    for (long i = 0; i <= 2000; i++) {
        throttle = random_val;
        esc1.setThrottle(throttle);
        Serial.println(throttle);
        delay(5);
    }

    /* 
    for (int throttle = min_throttle; throttle < max_throttle; throttle += 10) {
       esc1.setThrottle(throttle);
       Serial.println(throttle);
       delay(5);
    }
    for (int throttle = max_throttle; throttle > min_throttle; throttle -= 10) {
       esc1.setThrottle(throttle);
       Serial.println(throttle);
       delay(5);
    } */

    /* Descend to zero */
    for (long i = 5; i > 0; i--) {
        throttle = i * (random_val / 5);
        esc1.setThrottle(throttle);
        Serial.println(throttle);
        delay(5);
    }

    esc1.setThrottle(48);
    for (long i = 0; i < 400; i++) {
        Serial.println(48);
        delay(5);  
    }
    //delay(2000);
}
