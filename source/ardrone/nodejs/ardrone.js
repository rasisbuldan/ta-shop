var fs = require('fs')

var arDrone = require('ar-drone');
var client = new arDrone.createClient();

/* Config : send all navdata */
client.config('general:navdata_demo', 'FALSE');

/* Global variable */
var t0 = 0;
var t1 = 0;

console.log('Connected.. collecting navigation data');
flight_time = new Date().getTime();
t0 = new Date().getTime();

/* Calibrate FTRIM before flight */
//client.calibrate(1);

/*
    Flight plan:
        - Takeoff (4s)
        - Front (2s)
        - Up (2s)
        - Back (2s)
        - Left (2s)
        - Down (3s)
        - Land (3s)
*/

/* Hover for 10s */
try {
    client
        .after(4000, function() {
            client.takeoff();
        })

        .after(10000, function() {
            client.stop();
            client.land();
        });
}

/* Ascend */
/* try {
    client
        .after(4000, function() {
            client.takeoff();
        })

        .after(6000, function() {
            this.up(0.3);
        })

        .after(4000, function() {
            this.down(0.3);
        })

        .after(4000, function() {
            this.stop();
            this.land();
        });
} */

/* Maneuver */
/* try {
    client
        .after(4000, function() {
            client.takeoff();
        })

        .after(6000, function() {
            this.front(0.1);
        })

        .after(4000, function() {
            this.up(0.1);
        })

        .after(4000, function() {
            this.back(0.15);
        })

        .after(4000, function() {
            this.down(0.1);
        })

        .after(4000, function() {
            this.stop();
            this.land();
        });
} */

catch(err) {
    console.log("Error: ", err)
    client
        .after(100, function() {
            this.stop();
            this.land();
        });
}

/* Navigation data received */
i = 0;
client.on('navdata', (navdata) => {
    i++;
    t1 = new Date().getTime();
    if (navdata.demo) {
        navdata.timeDelta = t1 - t0;
        filename = 'flight-data/flight_' + flight_time + '.json'
        fs.appendFile(filename, JSON.stringify(navdata) + "\n", () => {
            console.log('Flight data point: ' + i);
        });
    }
    t0 = new Date().getTime();
});