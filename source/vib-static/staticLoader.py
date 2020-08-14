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
stored_filename_path = 'C:/Users/rss75/Documents/GitHub/ta-shop/source/vib-static/stored_v3.txt'
data_path = 'C:/Users/rss75/Documents/GitHub/ta-shop/source/data/static_pwm_to_db_aug6'

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
    # Check if filename already loaded
    with open(stored_filename_path) as stored_file:
        stored = stored_file.readlines()

        # Filename already stored
        if (filename + '\n') in stored:
            print('[!] File {} already stored!'.format(filename))
            return

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

                pwmVal = int(data[1])
                if pwmVal == 50:
                    payload['pwm'] = 100
                else:
                    payload['pwm'] = pwmVal

            
            except Exception as e:
                print('Data:', data)
                traceback.print_exc()
                sys.exit()

            # Insert into MongoDB
            idNum = staticpwmCollection.insert_one(payload)

    # Add to stored filename document
    with open(stored_filename_path, 'a+') as stored_file:
        stored_file.write(filename+'\n')
    
    print('Load {} with {} data points complete!'.format(filename, i))


### Main ###
# MongoDB Connector
clientDB = pymongo.MongoClient("localhost", 27017)
staticpwmCollection = clientDB['test-db']['staticpwm3s']

dirlist = getDirectoryList(data_path)
for file in dirlist:
    loadDataset(file)