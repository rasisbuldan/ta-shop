import pymongo
import numpy as np
import time
import os
import sys

class FlightData:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.initializeConnectionDB()


    def initializeConnectionDB(self, host='localhost', port=27017):
        ''' Connect to database with default localhost '''
        self.clientDB = pymongo.MongoClient(host, port)

        self.flightCollection = self.clientDB['test-db']['flightdatas']
        if self.verbose:
            print('Connection to test-db.flightdatas successful with', self.numOfDocuments(), 'documents')


    def numOfDocuments(self):
        ''' Number of all documents in collection '''
        return self.flightCollection.count_documents({})


    def numOfDocumentsByDesc(self, desc):
        ''' Number of documents by description '''
        return self.flightCollection.count_documents({
            "description": {"$eq" : desc}
        })


    def getFlightTimeByDesc(self, desc, landed=True):
        if landed:
            tsArray = [n['timestamp'] for n in list(self.flightCollection.find({
                "description": { "$eq": desc }
            },
            {
                "_id": 0,
                "timestamp": 1
            }))]

        else:
            tsArray = [n['timestamp'] for n in list(self.flightCollection.find({
                "$and" : [
                    {
                        "description": { "$eq": desc }
                    },
                    {
                        "state.controlState": { "$ne": "CTRL_LANDED" }
                    },
                ]
            },
            {
                "_id": 0,
                "timestamp": 1
            }))]

        return max(tsArray) - min(tsArray)


    def listDescription(self):
        ''' Unique description list in collection '''
        return list(self.flightCollection.distinct('description'))
    

##################
##### Driver #####
##################

if __name__ == '__main__':
    FD = FlightData()
    
    listDesc = FD.listDescription()
    #print('Description List')
    #print(*listDesc, sep='\n')
    #print('\n--------------------\n')
    
    includeDesc = ['aug11_6'] # , 'aug11'
    excludeDesc = ['fail', 'up_down', 'test', 'crash']

    totalNumOfDesc = 0
    totalNumOfDocs = 0
    totalFlightTime = 0
    for desc in listDesc:

        # Selected 
        if not any(fd in desc for fd in includeDesc) or any(fd in desc for fd in excludeDesc):
            continue
        else:
            totalNumOfDesc += 1
            numOfDocs = FD.numOfDocumentsByDesc(desc)
            flightTime = FD.getFlightTimeByDesc(desc)
            print('{}: {} documents in {:.2f}s'.format(desc, numOfDocs, flightTime/1000))
            totalNumOfDocs += numOfDocs
            totalFlightTime += flightTime
    
    print('Total number of documents: {} in {} attempts in {:.2f}s'.format(totalNumOfDocs, totalNumOfDesc, totalFlightTime/1000))