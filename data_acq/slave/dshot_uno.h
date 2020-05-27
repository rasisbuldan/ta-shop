/************************************************/
/************************************************/
/************* DShot Communicaton ***************/
/************************************************/
/************************************************/
/*      DShot600 for 16MHz CPU (Arduino Uno)    */
/*      Bit banging method                      */
/*      T0H: 10 cycles                          */
/*      T0L: 17 cycles                          */
/*      T1H: 20 cycles                          */
/*      T1L: 7  cycles                          */
/*      -----------------------------------     */
/*           27 cycles per bit                  */
/************************************************/
/************************************************/

#include "Arduino.h"

#ifndef dshot_uno
#define dshot_uno

/* Define PortD (pin D0-D7) on Arduino Uno */
#define DSHOT_PORT PORTD

/**** Dshot Class ****/
class DShot {
   public:
    DShot();
    void attach(uint8_t pin);
    uint16_t setThrottle(uint16_t throttle);

   private:
    uint16_t _packet = 0;
    uint16_t _throttle = 0;
    uint8_t _pinMask = 0;
};

#endif