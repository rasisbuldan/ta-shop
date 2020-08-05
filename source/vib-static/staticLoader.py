'''
    NavData load from json file to MongoDB
    using folder file traversal 'data_path'
'''

import numpy as np
import pymongo
import sys
import os
import traceback

### Variable ###
data_path = 'C:/Users/rss75/Documents/GitHub/ta-shop/source/data/static_pwm_to_db_aug5/v2'

dataTemplate = {
    "description": "",
    "timestart": 0,
    "timestamp": 0,
    "pwm": 0
}

def getDataFromFile(filename):
    '''
        Retreive data from file as array
    '''
    filearr = np.genfromtxt(filename, delimiter=",", dtype='int64')
    return filearr


def getDirectoryList(data_dir):
    '''
        Get list of file in directory
    '''
    return [f for f in os.listdir(data_dir) if '.txt' in f]


def loadDataset(filename):
    '''
        File traversal, load into database via dictionary
    '''
    # Filename parse
    fn = filename.split("_")
    timestart = int(fn[3])
    desc = '_'.join(fn[4:]).replace('.txt','')

    # Load raw navdata from file
    rawData = getDataFromFile(data_path + '/' + filename)
    
    # Populate template with value from file
    i = 0
    timestartBatch = timestart
    batchNum = 0
    rawDataLength = len(rawData)

    for data in rawData:
        if data[1] == 5000:
            batchNum += 1
            timestartBatch = int(data[0])
        else:
            i += 1
            print('Loading [{}/{}] {:.1f}%'.format(i, rawDataLength, i/rawDataLength*100), end='\r')
            try:
                payload = dataTemplate.copy()
                payload['description'] = desc + '_' + str(batchNum)
                payload['timestart'] = timestartBatch
                payload['timestamp'] = int(data[0])
                payload['pwm'] = int(data[1])

            
            except Exception as e:
                print('Data:', data)
                traceback.print_exc()
                sys.exit()

            # Insert into MongoDB
            idNum = staticpwmCollection.insert_one(payload)
    
    print('Load {} with {} data points complete!'.format(filename, i))


### Main ###
# MongoDB Connector
clientDB = pymongo.MongoClient("localhost", 27017)
staticpwmCollection = clientDB['test-db']['staticpwm2s']

dirlist = getDirectoryList(data_path)
for file in dirlist:
    loadDataset(file)