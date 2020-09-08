import matplotlib.pyplot as plt
import pickle
import os

filename = 'D:/mc/mc_model_20_09_05_02_59_02.h5_1000_aug9_0_aug10_1_aug10_2_aug10_3_aug11_4_aug11_5_aug11_6.pkl'

with open(filename, 'rb') as historyFile:
    history = pickle.load(historyFile)

#print(len(history['val_accuracy']))
#print(history.keys())
fig = plt.figure(figsize=(16,10))
p1, = plt.plot(history['accuracy'], linewidth=2)
p2, = plt.plot(history['val_accuracy'], linewidth=2)
plt.grid(True)
plt.ylabel('Akurasi', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.yticks([acc/100 for acc in range(45,101,5)])
plt.legend(
    (p1,p2),
    ('Akurasi Pelatihan', 'Akurasi Pengujian'),
    fontsize=16
)
plt.show()