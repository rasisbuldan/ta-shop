/* DShot600 for 16MHz CPU (Arduino Uno)
 *      T0H: 10 cycles
 *      T0L: 17 cycles
 *      T1H: 20 cycles
 *      T1L: 7  cycles
 * ---------------------------
 *           27 cycles per bit
 */

#include "Arduino.h"

/* Define PortD (pin D0-D7) on Arduino Uno */
#define DSHOT_PORT PORTD

/* Assembly macros */
#define NOP "NOP\n"
#define NOP2 NOP NOP
#define NOP4 NOP2 NOP2
#define NOP8 NOP4 NOP4

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

/* Packet array */
static uint8_t dShotBits[16];

/* Pin attached to DShot */
static uint8_t dShotPins = 0;

/* State variable if ISR is initialized */
static boolean timerActive = false;

static boolean isTimerActive(){
    return timerActive;
}


/* Sending data (assembly) */
static inline void sendData(){
    /* Set no interrupts allowed on data sending */
    noInterrupts();

    asm(
        "LDI r23, 0 \n"     // for i=0; i<=15; i++
        "IN r25, %0 \n"     // Set HIGH for attached pins (DSHOT_PORT |= dShotPins)
        "_for_loop: \n"
        "OR r25, %1 \n"
        "OUT %0, r25 \n"
        NOP4                // hold for 10 cycles
        NOP2
        "LD r24, Z+ \n"     // Set LOW for low bits only (DSHOT_PORT &= dShotBits[i])
        "AND r25, r24 \n"
        "OUT %0, r25 \n"
        NOP8                // hold for 10 cycles
        "AND r25, %2 \n"    // Set LOW everything (DSHOT_PORT &= ~dShotPins)
        "OUT %0, r25 \n"
        "INC r23 \n"        // add to i (tmp_reg)
        "CPI r23, 16 \n"
        "BRLO _for_loop \n" // 7 cycles to next bit (4: add to i, 2: turn on)
        :
		: "I" (_SFR_IO_ADDR(DSHOT_PORT)), "r" (dShotPins), "r" (~dShotPins), "z" (dShotBits)
		: "r25", "r24", "r23"
    );

    /* Allow interrupts */
    interrupts();
}


/* Set Arduino ISR timer interrupts function -> ISR(TIMER1_COMPA_vect)*/
/* Timer0 for frequency 1000Hz */
/* ps: Timer0 used by millis() and delay() */
static void initISR(){
    cli(); // stop interrupts
    TCCR1A = 0; // set entire TCCR1A register to 0
    TCCR1B = 0; // set entire TCCR1B register to 0
    TCNT1  = 0; // initialize counter value to 0
    // set compare match register for 1000 Hz increments
    OCR1A = 15999; // = 16000000 / (1 * 1000) - 1 (must be <65536)
    // turn on CTC mode
    TCCR1B |= (1 << WGM12);
    // Set CS12, CS11 and CS10 bits for 1 prescaler
    TCCR1B |= (0 << CS12) | (0 << CS11) | (1 << CS10);
    // enable timer compare interrupt
    TIMSK1 |= (1 << OCIE1A);
    
    // Reset packet content?
    timerActive = true;
    for (byte i = 0; i < 16; i++){
        dShotBits[i] = 0;
    }
    dShotPins = 0;

    sei(); // allow interrupts
}

/* Send data when ISR triggered */
ISR(TIMER1_COMPA_vect){
    sendData();
}


/* Create DShot data packet */
/* 11-bit throttle data, 1-bit telemetry request bit, 4-bit checksum CRC */
static inline uint16_t createPacket(uint16_t throttle){
    /* Initialize CRC */
    uint8_t crc = 0;

    throttle <<=1;
    
    /* Set throttle value to minimum if less than 48 */
    if (throttle < 48){
        throttle |= 1;
    }

    /* Calculate CRC based on throttle data */
    uint16_t crc_data = throttle;
    for (byte i = 0; i < 3; i++){
        crc ^= crc_data;
        crc_data >>= 4;
    }
    crc &= 0x000F;
    
    return (throttle << 4) | crc;
}


/* DShot object declaration */
DShot::DShot(){}

/* Attach pin to DShot */
void DShot::attach(uint8_t pin){
    /* Store class attribute */
    this->_packet = 0;
    this->_pinMask = digitalPinToBitMask(pin);
    
    /* Set pinmode */
    pinMode(pin, OUTPUT);

    /* Initialize ISR if not initialized */
    if (!isTimerActive()){
        initISR();
    }
    dShotPins |= this->_pinMask;
}

/* Set data packet throttle */
uint16_t DShot::setThrottle(uint16_t throttle){
    /* Store class attribute */
    this->_throttle = throttle;

    this->_packet = createPacket(throttle);
    uint16_t mask = 0x8000;

    for (byte i = 0; i < 16; i++){
        if (this->_packet & mask){
            dShotBits[i] |= this->_pinMask;
        }
        else{
            dShotBits[i] &= ~(this->_pinMask);
        }
        mask >>= 1;
    }

    return _packet;
}


/*********************************************/
/*********************************************/
/***************Main Program******************/
/*********************************************/
/*********************************************/
/* To do:  
 *      - Serial communication to master pin
 *      - Sweep, random throttle
 */

uint16_t throttle = 0;
uint16_t target = 0;
uint16_t init_throttle = 48;
uint16_t min_throttle = 800;
uint16_t max_throttle = 900;
uint16_t const_throttle = 400;

DShot esc1;

void setup(){
    Serial.begin(115200);
    Serial.println("Initializing ESC...");
    
    /* Attach ESC to pin D7 */
    esc1.attach(7);
    
    /* Set initial throttle */
    esc1.setThrottle(init_throttle);
}

void loop(){
    /* Increasing throttle */
    for (throttle = min_throttle; throttle <= max_throttle; throttle++){
        esc1.setThrottle(throttle);
        delay(20);
    }

    /* Decreasing throttle */
    for (throttle = max_throttle; throttle >= min_throttle; throttle--){
        esc1.setThrottle(throttle);
        delay(20);
    }
}