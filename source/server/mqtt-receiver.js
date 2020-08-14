/********** Initialization **********/
/* Aedes broker */
const aedes = require('aedes')();
const server = require('net').createServer(aedes.handle);
const brokerPort = 1884;

server.listen(brokerPort, () => {
  console.log(`[Aedes] Broker started and listening to port ${brokerPort}`);
});

aedes.on('client', (client) => {
  console.log(`[Aedes] Client connected with id ${client.id}`);
});

aedes.on('subscribe', (subscriptions, client) => {
  console.log(`[Aedes] Client with id ${client.id} subscribed with topic ${subscriptions[0].topic}`);
});

aedes.on('clientDisconnect', (client) => {
  console.log(`[Aedes] Client disconnected with id ${client.id}`);
});


/* MQTT Initialization */
var subscribeTopic = [
  'topic/pi/mpu6050',
  'topic/navdataraw'
];
const mqttHost = 'localhost';
const mqttPort = 1884;
const mqtt = require('mqtt');
const mqttClient = mqtt.connect(`mqtt://${mqttHost}:${mqttPort}`);


/* MongoDB Connector */
const dbHost = 'localhost';
const dbName = 'test-db';
const mongoose = require('mongoose');
require('mongoose-long')(mongoose);

var SchemaTypes = mongoose.Schema.Types;

mongoose.connect(`mongodb://${dbHost}/${dbName}`, {useNewUrlParser: true, useUnifiedTopology: true});
const db = mongoose.connection;

db.on('error', (err) => {
  console.log('Error: ', err.message);
});

db.once('open', () => {
  console.log(`[DB] Connection to ${dbHost} successful!`);
});


/* Schema definition */
const accelDataSchema = new mongoose.Schema({
  timestamp: {type: Number},
  mpu1: {
    x: {type: Number},
    y: {type: Number},
    z: {type: Number}
  },
  mpu2: {
    x: {type: Number},
    y: {type: Number},
    z: {type: Number}
  }
},
{
  versionKey: false
});

const AccelData = mongoose.model('AccelData', accelDataSchema);

/********** MQTT **********/
/* Connect to broker and subscribe */
mqttClient.on('connect', () => {
  console.log(`[MQTT] Connection to ${mqttHost} successful!`);
  /* Batch subscribe */
  for (topic of subscribeTopic) {
    mqttClient.subscribe(topic, (err, granted) => {
      if (!err) {
        console.log(`[MQTT] Subscription to ${granted[0].topic} at qos ${granted[0].qos} successful!`);
      }
      else {
        console.log('Error occured: ', err);
      }
    });
  }
});


mqttClient.on('message', (topic, message) => {
  let d = new Date();
  let dateString = d.toLocaleString();

  /* MPU6050 raw vibration (2 mpu) - Raspi ZeroW */
  if (topic == 'topic/pi/mpu6050') {
    let msgData = JSON.parse(message.toString());

    /* Insert to DB */
    AccelData.create({
      timestamp: Math.floor(Number(msgData.timestamp)/1000000),
      mpu1: {
        x: Number(msgData.mpu1.x),
        y: Number(msgData.mpu1.y),
        z: Number(msgData.mpu1.z)
      },
      mpu2: {
        x: Number(msgData.mpu2.x),
        y: Number(msgData.mpu2.y),
        z: Number(msgData.mpu2.z)
      }
    });
  }

  /* Navigation data (ARDrone) */
  else if (topic == 'topic/navdataraw') {
    let navdata = JSON.parse(message.toString());
    let payload = {
      navdata: {
        batteryPercentage: 0,
        altitude: 0,
        orientation: {
          roll: 0,
          pitch: 0,
          yaw: 0,
        },
        pwm: {
          mot1: 0,
          mot2: 0,
          mot3: 0,
          mot4: 0,
        },
      }
    }
  
    if (navdata.demo) {
      payload.navdata.batteryPercentage = navdata.demo.batteryPercentage
      payload.navdata.orientation.roll = navdata.demo.leftRightDegrees
      payload.navdata.orientation.pitch = navdata.demo.frontBackDegrees
      payload.navdata.orientation.yaw = navdata.demo.clockwiseDegrees
      payload.navdata.altitude = navdata.demo.altitude
    }
  
    if (navdata.pwm) {
      payload.navdata.pwm.mot1 = navdata.pwm.motors[0]
      payload.navdata.pwm.mot2 = navdata.pwm.motors[1]
      payload.navdata.pwm.mot3 = navdata.pwm.motors[2]
      payload.navdata.pwm.mot4 = navdata.pwm.motors[3]
    }
    
    let navdataTimeMs = d.getTime();
    NavdataDrone.create({
      timestamp: navdataTimeMs.toString(),
      navdata: payload.navdata
    });
  }
});