var arDrone = require('ar-drone');
const { fstat } = require('fs');
var client = new arDrone.createClient();
client.config('general:navdata_demo', 'FALSE');
var fs = require('fs')
var navdataBuf = []
var t0 = 0;
var t1 = 0;

/* function saveToFile(data) {
    fs.writeFile('./data.txt', JSON.stringify(data), 'utf-8')
} */

console.log('Connected.. collecting navigation data');
t0 = new Date().getTime();
client.calibrate(1)
/* client.takeoff();

client
    .after(5000, function(){
        this.clockwise(0.5);
    })
    .after(3000, function(){
        this.stop();
        this.land();
    }) */

//
client.on('navdata', (navdata) => {
    t1 = new Date().getTime();
    //console.log('Interpacket time ', t1 - t0);
    //console.log('Update rate ', 1000 / (t1 - t0))
    if (navdata.demo) {
        navdata.timeDelta = t1 - t0;
        //console.log('navdata: ', navdata);
        /* const navdataObj = {
            batteryPercentage: navdata.demo.batteryPercentage,
            roll: navdata.demo.rotation.roll,
            yaw: navdata.demo.rotation.yaw,
            pitch: navdata.demo.rotation.pitch,
            altitudeMeters: navdata.demo.altitudeMeters,
            xVelocity: navdata.demo.velocity.x,
            yVelocity: navdata.demo.velocity.y,
            zVelocity: navdata.demo.velocity.z,
        } */
        //console.log('navdata: ', navdataObj)
        //navdataBuf.push(navdataObj)
        //navdataBuf.push(navdata.demo);
        fs.appendFile('data.txt', JSON.stringify(navdata) + "\n", () => {
            console.log('Appending data', navdataBuf.length);
        });
    }
    t0 = new Date().getTime();
});