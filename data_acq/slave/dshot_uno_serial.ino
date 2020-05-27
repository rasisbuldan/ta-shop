/* Arduino slave for sending data to ESC */
/* Serial bidirectional to Raspberry Pi (master) at 115200bps */
/* Current: 2 ESC support */
/* To do:
 *      - JSON payload
 */

/*********************************************/
/*********************************************/
/*************** Main Program ****************/
/*********************************************/
/*********************************************/

#include "dshot_uno.h"

boolean verbose = 0;

/* Global variable declaration */
String payload;
char pChar;
struct droneState {
    boolean powerState;
    String operatingState;
    uint16_t esc1Speed;
    uint16_t esc2Speed;
};
uint16_t esc1_throttle, esc2_throttle;

/* Object declaration */
droneState dstate;
DShot esc1;
DShot esc2;

void parsePayload(String p, droneState* ds){
    /* Payload structure */
    /* [0]      : power_state (0/1) */
    /* [1-3]    : operating_state (hover, statc, ascnd, descn, ) */
    /* [3-6]    : esc1 speed (48-2047) */
    /* [7-10]   : esc2 speed (48-2047) */

    ds->powerState = p.substring(0,1);
    ds->operatingState = p.substring(1,4);
    ds->esc1Speed = p.substring(4,8).toInt();
    ds->esc2Speed = p.substring(8,12).toInt();
}

void setup(){
    /* Serial initialization */
    Serial.begin(115200);
    Serial.println("Serial ready!");

    /* ESC pin initialization */
    esc1.attach(7);     // D7
    esc1.attach(8);     // D8

    /* ESC start-up command */
    esc1.setThrottle(48);
    esc2.setThrottle(48);

    /* Variable initialization */
    payload = "";
}

void loop(){
    /* Incoming message from serial */
    if (Serial.available() > 0){
        if (verbose){
            Serial.print("Incoming serial!");
        }

        /* Read inline payload */
        payload = Serial.readStringUntil("\n");
        
        /* Debug payload */
        if(verbose){
            Serial.print("Received payload: ");
            Serial.println(payload);
        }
    }

    /* Parse incoming payload (fixed length message (12-byte)) */
    parsePayload(payload, &dstate);

    /* Set throttle for each ESC */
    esc1_throttle = esc1.setThrottle(dstate.esc1Speed);
    esc2_throttle = esc2.setThrottle(dstate.esc2Speed);

    if (verbose){
        Serial.print("esc1: ");
        Serial.print(esc1_throttle);
        Serial.print(" \t esc2: ");
        Serial.println(esc2_throttle);
    }

    /* Update speed adjustment (to be tested) */
    delay(20);
}