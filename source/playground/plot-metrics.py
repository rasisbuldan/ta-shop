import matplotlib.pyplot as plt
import os
import pickle

def plotMetricsHistory(history):
    '''
        Plot metrics (currently supported: mean squared error)
    '''
    # Set font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    plotData = history['loss']
    plotDataVal = history['val_loss']

    p1, = plt.plot(plotData, 'r-')
    p2, = plt.plot(plotDataVal, 'k-')
    plt.ylim(0,50)
    plt.xlim(0,200)
    plt.yticks([i for i in range(0,51,5)])
    plt.grid(True)
    plt.legend(
        (p1,p2),
        ('Training MSE', 'Validation MSE'),
        loc='upper right',
        fontsize=18,
    )



def loadHistory(filename):
    with open(filename, 'rb') as historyFile:
        history = pickle.load(historyFile)
    return history


if __name__ == '__main__':
    history = loadHistory('D:/Cloud/Google Drive/Tugas Akhir/data/cache/Aug21/lstm_navdatavib_model_multidenseoutfilter_aug20_20_08_25_01_25_06_300_aug9_0_aug7_2_jul29_2.pkl')

    plotMetricsHistory(history)
    plt.show()