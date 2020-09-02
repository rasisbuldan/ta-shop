import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model

model = load_model('D:/Cloud/Google Drive/Tugas Akhir/data/cache/Aug21/plot/20_08_25_01_27_10_300_selected/lstm_navdatavib_model_multidenseoutfilter_aug20_20_08_25_01_25_06_300_aug9_0_aug7_2_jul29_2.h5')


import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, 'source/ardrone/modeljs')