import numpy as np
from tabulate import tabulate

def RMSE(arr_true, arr_pred):
    '''
        Calculate Root Mean Squared Error
    '''
    if len(arr_true) != len(arr_pred):
        print('Array length not equal, value of RMSE can\'t be calculated')
        return
    
    nData = len(arr_true)

    return np.sqrt(
        np.sum(
            np.square(
                np.subtract(arr_true, arr_pred))
        )/nData)

def SMAPE(arr_true, arr_pred):
    '''
        Calculate Symmetric Mean Absolute Percentage Error
    '''
    if len(arr_true) != len(arr_pred):
        print('Array length not equal, value of RMSE can\'t be calculated')
        return

    nData = len(arr_true)

    return (
        100/nData 
        * np.sum(2 
            * np.abs(arr_true - arr_pred) 
            / (np.abs(arr_true) + np.abs(arr_pred))))


def RSquared(arr_true, arr_pred):
    '''
        Calculate R^2 (Coefficient of Determination)
    '''
    if len(arr_true) != len(arr_pred):
        print('Array length not equal, value of RMSE can\'t be calculated')
        return

    arrMean = np.mean(arr_true)

    numerator = np.sum(np.square(np.subtract(arr_true, arr_pred)))
    denominator = np.sum(np.square(np.subtract(arr_true, arrMean)))

    return (1 - numerator/denominator)


def MAE(arr_true, arr_pred):
    '''
        Calculate Mean Absolute Error
    '''
    if len(arr_true) != len(arr_pred):
        print('Array length not equal, value of RMSE can\'t be calculated')
        return
    
    nData = len(arr_true)

    return (1/nData) * np.sum(np.abs(np.subtract(arr_true, arr_pred)))


def printMetrics(arr, filename=None):
    '''
        Pretty print using tabulate (with format 'pretty')
    '''
    if filename != None:
        arr = np.loadtxt(filename, delimiter=',')

    featureName = [
        'RMS (X)',
        'RMS (Y)',
        'RMS (Z)',
        'Kurtosis (X)',
        'Kurtosis (Y)',
        'Kurtosis (Z)',
        'Skewness (X)',
        'Skewness (Y)',
        'Skewness (Z)',
        'Crest Factor (X)',
        'Crest Factor (Y)',
        'Crest Factor (Z)',
        'Peak-To-Peak (X)',
        'Peak-To-Peak (Y)',
        'Peak-To-Peak (Z)',
    ]
    header = ['Feature', 'RMSE', 'SMAPE', 'MAE', 'RSquared']
    data = [[featureName[i]] + list(arr[i,:]) for i in range(arr.shape[0])]
    rmseVal = [data[i][1] for i in range(len(data))]
    print(np.mean(rmseVal))

    print(tabulate(
        tabular_data=data, 
        headers=header, 
        tablefmt='pretty',
        floatfmt=".4f"
    ))
    
    



### Driver ###
if __name__ == '__main__':
    printMetrics(
        arr=[],
        filename='D:/Cloud/Google Drive/Tugas Akhir/data/cache/Aug21/metrics/metrics_20_08_24_14_21_29_200.txt')
