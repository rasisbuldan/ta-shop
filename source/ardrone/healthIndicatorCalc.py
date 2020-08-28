import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os
import sys

offset = [127, 26, 49, 26, 25, 37, 28]

##### Real Data #####
dataPath = 'D:/Cloud/Google Drive/Tugas Akhir/data/dumps/seq_agg_load5'
dirList = os.listdir(dataPath)
filename = [fn for fn in dirList if '.csv' in fn][0]
featureData = pd.read_csv(os.path.join(dataPath, filename))
featureDataSelectedArr = featureData[['rms1y', 'kurt1y', 'skew1y', 'crest1y']]
print(filename, len(featureDataSelectedArr))

for filename in [fn for fn in dirList if '.csv' in fn][1:]:
    featureData = pd.read_csv(os.path.join(dataPath, filename))
    featureDataSelected = featureData[['rms1y', 'kurt1y', 'skew1y', 'crest1y']]
    featureDataSelectedArr = featureDataSelectedArr.append(featureDataSelected, ignore_index=True)
    print(filename, len(featureDataSelected))


# Calculate PCA
print(len(featureDataSelectedArr))
pca = PCA(n_components=1)
pca.fit(featureDataSelectedArr)
featureDataSelectedArr['HI'] = pca.transform(featureDataSelectedArr)
featureDataSelectedArrEWM = featureDataSelectedArr.ewm(alpha=0.05).mean()
featureDataSelectedArr['HIs'] = featureDataSelectedArrEWM['HI']

scaler = MinMaxScaler()
scaler.fit(featureDataSelectedArr['HIs'].to_numpy().reshape(-1,1))
featureDataSelectedArr['HIss'] = scaler.transform(featureDataSelectedArr['HIs'].to_numpy().reshape(-1,1))

featureDataSelectedArr.to_csv('realdata.csv')

realData = featureDataSelectedArr
realDataEWM = featureDataSelectedArrEWM



##### Predicted Data #####
dataPath = 'D:/Cloud/Google Drive/Tugas Akhir/data/dumps/seq_agg_load_pred'
dirList = os.listdir(dataPath)

filename = [fn for fn in dirList if '.csv' in fn][0]
featureData = pd.read_csv(os.path.join(dataPath, filename))
featureDataSelectedArr = featureData[['rms1y','kurt1y', 'skew1y', 'crest1y']]
print(filename, len(featureDataSelectedArr))

i = 1
for filename in [fn for fn in dirList if '.csv' in fn][1:]:
    featureData = pd.read_csv(os.path.join(dataPath, filename))
    featureDataSelected = featureData[['rms1y','kurt1y', 'skew1y', 'crest1y']]
    
    featureDataOffset = featureDataSelectedArr.iloc[[len(featureDataSelectedArr)-1]]
    for _ in range(offset[i]):
        featureDataOffset = featureDataOffset.append(featureDataSelectedArr.iloc[[len(featureDataSelectedArr)-1]], ignore_index=True)
    
    featureDataSelected = featureDataOffset.append(featureDataSelected, ignore_index=True)

    featureDataSelectedArr = featureDataSelectedArr.append(featureDataSelected, ignore_index=True)
    print(filename, len(featureDataSelected))
    i += 1


# Calculate PCA
print(len(featureDataSelectedArr))
pca = PCA(n_components=1)
pca.fit(featureDataSelectedArr)
featureDataSelectedArr['HI'] = pca.transform(featureDataSelectedArr)
featureDataSelectedArrEWM = featureDataSelectedArr.ewm(alpha=0.05).mean()
featureDataSelectedArr['HIs'] = featureDataSelectedArrEWM['HI']

scaler = MinMaxScaler()
scaler.fit(featureDataSelectedArr['HIs'].to_numpy().reshape(-1,1))
featureDataSelectedArr['HIss'] = scaler.transform(featureDataSelectedArr['HIs'].to_numpy().reshape(-1,1))

featureDataSelectedArr.to_csv('predicteddata.csv')

predictedData = featureDataSelectedArr
predictedDataEWM = featureDataSelectedArrEWM



##### Data Plotting #####
fig = plt.figure()

p1, = plt.plot(list(range(len(realData))), realData['rms1y'], 'r--')
p2, = plt.plot(list(range(offset[0],len(predictedData)+offset[0])), predictedData['rms1y'], 'k-')
plt.title('RMS')
plt.xlim(0,500)
plt.grid(True)
plt.legend((p1,p2),('Real Data', 'Predicted Data'), fontsize=18)
plt.show()

fig = plt.figure()

p1, = plt.plot(list(range(len(realData))), realData['skew1y'], 'r--')
p2, = plt.plot(list(range(offset[0],len(predictedData)+offset[0])), predictedData['skew1y'], 'k-')
plt.title('Skewness')
plt.xlim(0,500)
plt.grid(True)
plt.legend((p1,p2),('Real Data', 'Predicted Data'), fontsize=18)
plt.show()

fig = plt.figure()

p1, = plt.plot(list(range(len(realData))), realData['kurt1y'], 'r--')
p2, = plt.plot(list(range(offset[0],len(predictedData)+offset[0])), predictedData['kurt1y'], 'k-')
plt.title('Kurtosis')
plt.xlim(0,500)
plt.grid(True)
plt.legend((p1,p2),('Real Data', 'Predicted Data'), fontsize=18)
plt.show()

fig = plt.figure()

p1, = plt.plot(list(range(len(realData))), realData['crest1y'], 'r--')
p2, = plt.plot(list(range(offset[0],len(predictedData)+offset[0])), predictedData['crest1y'], 'k-')
plt.title('Crest Factor')
plt.xlim(0,500)
plt.grid(True)
plt.legend((p1,p2),('Real Data', 'Predicted Data'), fontsize=18)
plt.show()



##### Data Plotting #####
fig = plt.figure()

p1, = plt.plot(list(range(len(realData))), realData, 'r--')
p2, = plt.plot(list(range(offset[0],len(predictedData)+offset[0])), predictedData, 'k-')
plt.grid(True)
plt.legend((p1,p2),('Real Data', 'Predicted Data'), fontsize=18)
plt.show()