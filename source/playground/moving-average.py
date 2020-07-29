import numpy as np
import matplotlib.pyplot as plt
from random import random, seed

""" def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

x = np.arange(10000)
y = np.random.rand(10000)
N = 100

plt.plot(x, y)
plt.plot(x, running_mean(y,N))
plt.show() """


################
def genSineNoise(n_data):
    sineArray = np.sin(np.arange(n_data))


def genTrendNoise(n_data):
    trendData = []
    seed(12341234)
    for i in range(n_data):
        trendData.append(i + 0.5*i*random() - 2*i*random() + i*random())

    return trendData

def calcRMS(window_arr):
    N = len(window_arr)
    return np.sqrt(np.sum(np.square(window_arr))/N)


trendData = genTrendNoise(1000)
movingRMSData = []
movingAvgData = []
window = 5
for i in range(0,1000,window):
    movingRMSData.append(calcRMS(trendData[i:i+window]))
    movingAvgData.append(np.mean(trendData[i:i+window]))
print(len(movingRMSData))

""" plt.plot(np.arange(1000), trendData)
plt.show() """
'''
fig = plt.figure()
ax1 = fig.add_subplot(111, label='1')
ax2 = fig.add_subplot(111, label='2', frame_on=False)
ax3 = fig.add_subplot(111, label='3', frame_on=False)

p1, = ax1.plot(np.arange(1000), trendData, color='C1')
ax1.set_ylim([0,2500])
p2, = ax2.plot(list(range(0,1000,window)), movingRMSData, color="C0", linewidth=2)
ax2.set_ylim([0,2500])
p3, = ax3.plot(list(range(0,1000,window)), movingAvgData, color="C2", linewidth=2)
ax3.set_ylim([0,2500])


ax1.legend((p1,p2,p3), ('Trend Data', 'Moving RMS', 'Moving Average'))
plt.grid(True)
plt.show() '''


################
''' Moving average with convolution '''
stepWeight = 0.1
windowSize = 10
weight = np.arange(stepWeight, stepWeight*(windowSize+1), stepWeight)
print('[weight]', weight)

trendData = genTrendNoise(1000)
trendDataSquared = np.square(trendData)
dataWeightedRMS = []

for d in range(windowSize, len(trendData)-windowSize):
    sumWeighted = 0
    for i in range(windowSize):
        sumWeighted += weight[i] * trendDataSquared[d-windowSize+i]
    
    dataWeightedRMS.append(np.sqrt(sumWeighted/np.sum(weight)))

    

fig = plt.figure()
ax1 = fig.add_subplot(111, label='1')
ax2 = fig.add_subplot(111, label='2', frame_on=False)
ax3 = fig.add_subplot(111, label='3', frame_on=False)
ax4 = fig.add_subplot(111, label='4', frame_on=False)

p1, = ax1.plot(np.arange(1000), trendData, color='C1')
ax1.set_ylim([0,2500])
p2, = ax2.plot(list(range(0,1000,window)), movingRMSData, color="C0", linewidth=1)
ax2.set_ylim([0,2500])
p3, = ax3.plot(list(range(0,1000,window)), movingAvgData, color="C2", linewidth=1)
ax3.set_ylim([0,2500])
p4, = ax4.plot(list(range(windowSize,1000-windowSize)), dataWeightedRMS, color="C3", linewidth=2)
ax4.set_ylim([0,2500])


ax1.legend((p1,p2,p3,p4), ('Trend Data', 'Moving RMS', 'Moving Average', 'Weighted RMS'))
plt.grid(True)
plt.show()