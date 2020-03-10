import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

var3 = getDataFile('result/var_3rd_test.txt')
kur3 = getDataFile('result/kurt_3rd_test.txt')
mean3 = getDataFile('result/mean_3rd_test.txt')
skew3 = getDataFile('result/skew_3rd_test.txt')
n3 = var3.shape[0]

# Preprocessing for specific test
buf = np.array([]).reshape(n1,0)
buf = np.append(buf, var1, axis=1)
buf = np.append(buf, kur1, axis=1)
buf = np.append(buf, mean1, axis=1)
buf = np.append(buf, skew1, axis=1)
z = np.zeros((n1_0,1), dtype=int)
o = np.ones((n1_1,1), dtype=int)
zo = np.append(z,o,axis=0)
ch1 = np.append(buf,zo,axis=1)
print('ch: ',str(1))
print(ch1)
print(ch1.shape)

# Separate to Input | Output
X = ch1[:,0:32].astype(float)
Y = ch1[:,32]

# Larger model
def create_larger():
    # Create model (neural network)
    model = Sequential()
    model.add(Dense(32, input_dim=32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

print('Estimating...')
# KerasClassifier
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=100, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Result: acc:%.2f%% stdev:(%.2f%%)" % (results.mean()*100, results.std()*100))
