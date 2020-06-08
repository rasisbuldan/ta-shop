/* DShot implementation on Arduino Uno */
/* with command from serial */
/* ESC attached to PORTB (D8-D13) */

#include <Arduino.h>
#include "DShot.h"

/* DShot instance declaration */
DShot esc;

/* Global variable */
String payload;
int throttleVal;


void setup(){
    Serial.begin(115200);
    Serial.print("Serial ready!");

    /* Attach DShot ESC to pin D9 */
    esc.attach(9);

    /* Arming sequence */
    throttleVal = 48;
    esc.setThrottle(48);
    delay(6000);
}

void loop(){
    /* Incoming message from serial */
    if (Serial.available() > 0){
        //Serial.println("Incoming serial!");

        /* Read inline payload (without newline char) */
        payload = Serial.readStringUntil("s");
        payload.remove(payload.length() - 1);
        throttleVal = payload.toInt();

        /* Check if payload throttle within range */
        if (throttleVal < 48){
            throttleVal = 48;
        }
        else if (throttleVal > 2047){
            throttleVal = 2047;
        }
        
        //Serial.print("Received payload: ");
        //Serial.println(payload);
        //Serial.println(payload.length());

        //Serial.print("Setting throttle to ");
        //Serial.println(throttleVal);
    }

    esc.setThrottle(throttleVal);
    delay(50);
}