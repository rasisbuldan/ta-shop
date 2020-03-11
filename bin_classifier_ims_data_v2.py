import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import random

# Retrieve data from file
def getDataFile(filename):
    filearr = np.genfromtxt(filename, delimiter=",", dtype='float')
    return filearr

# load dataset
var1 = getDataFile('result/var_1st_test.txt')
kur1 = getDataFile('result/kurt_1st_test.txt')
mean1 = getDataFile('result/mean_1st_test.txt')
skew1 = getDataFile('result/skew_1st_test.txt')
n1 = var1.shape[0]
n1_0 = int(0.85*n1)
n1_1 = n1 - n1_0
print(n1_0,' ',n1_1)

var2 = getDataFile('result/var_2nd_test.txt')
kur2 = getDataFile('result/kurt_2nd_test.txt')
mean2 = getDataFile('result/mean_2nd_test.txt')
skew2 = getDataFile('result/skew_2nd_test.txt')
n2 = var2.shape[0]
print(n2)

var3 = getDataFile('result/var_3rd_test.txt')
kur3 = getDataFile('result/kurt_3rd_test.txt')
mean3 = getDataFile('result/mean_3rd_test.txt')
skew3 = getDataFile('result/skew_3rd_test.txt')
n3 = var3.shape[0]

# Preprocessing for specific test
buf = np.array([]).reshape(n1,0)
buf = np.append(buf, var1[:,0:4], axis=1)
buf = np.append(buf, kur1[:,0:4], axis=1)
buf = np.append(buf, mean1[:,0:4], axis=1)
buf = np.append(buf, skew1[:,0:4], axis=1)
z = np.zeros((n1_0,1), dtype=int)
o = np.ones((n1_1,1), dtype=int)
zo = np.append(z,o,axis=0)
ch1 = np.append(buf,zo,axis=1)
print('ch: ',str(1))
print(ch1)
print(ch1.shape)

# Separate to Input | Output
X = ch1[:,0:16].astype(float)
Y = ch1[:,16]
print(Y)

print('Estimating...')
# Creating neural network
model = Sequential([
    keras.layers.Dense(16, input_dim=16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X, Y, epochs=30)
test_loss, test_acc = model.evaluate(X,  Y, verbose=2)
print('\nTest accuracy: {0:.3f} %'.format(float(test_acc*100)))

# Extract test data
test_data = [0,0,0,0,0,0,0,0,0,0]
ndata = [int(0.1*n2), int(0.2*n2), int(0.3*n2), int(0.4*n2), int(0.5*n2), int(0.6*n2), int(0.7*n2), int(0.8*n2), int(0.9*n2), int(n2-1)]
buf = np.array([]).reshape(n1,0)
buf = np.append(buf, var1[:,0:4], axis=1)
buf = np.append(buf, kur1[:,0:4], axis=1)
buf = np.append(buf, mean1[:,0:4], axis=1)
buf = np.append(buf, skew1[:,0:4], axis=1)
print(buf.shape)

# Testing classifier model
probability_model = Sequential([model])
for i in range(0,10):
    print(int(0+i*(0.1*n1)),' - ',int((0.1*n2)+i*(0.1*n1)))
    predictions = probability_model.predict(buf[int(0+i*(0.1*n1)):int((0.1*n1)+i*(0.1*n1)),:])
    print('predict at {0:.1f}'.format(0.1*i*100),' % of data: ',predictions[0])