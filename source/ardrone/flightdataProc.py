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


    def listDescription(self):
        ''' Unique description list in collection '''
        return list(self.flightCollection.distinct('description'))
    

##################
##### Driver #####
##################

if __name__ == '__main__':
    FD = FlightData()
    
    listDesc = FD.listDescription()
    print('Description List')
    print(*listDesc, sep='\n')