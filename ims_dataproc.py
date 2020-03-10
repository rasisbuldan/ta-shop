### To do ###
# 1. Convert filename to date
# 2. Entropy calculate probability distribution values

### Library ###
import time                             # Time
from datetime import timedelta          # Time formatting
import numpy as np                      # Array and matrix operation
import os                               # File operation
import scipy.stats as stats             # Statistical library
from io import StringIO                 # String handling
import matplotlib.pyplot as plt         # Plot operation
#import scipy.io                        # Matlab file handling
from scipy.optimize import curve_fit    # Curve fitting

# LSTM Implementation
import tensorflow as tf
import keras


### Global Variable ###
buf2 = np.empty([1, 8])
buf1 = np.array([])

### File extraction ###
def getData(filename):
    print("Processing " + filename)
    filearr = np.genfromtxt(filename, delimiter="\t", dtype='float')
    #for idx in range(0,filearr.shape[1]):
        #print("--------------------- idx: " + str(idx))
        #get_best_distribution(filearr[idx,:])
    data = getStats(filearr)
    return data

### Plot handling ###
def savePlot(data, testnum):
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    chnum = data.shape[1]//4
    i = 1
    for col in range(0, data.shape[1]):
        plt.plot(range(1, data.shape[0]+1), data[:, col], color=palette(col), label='ch'+str(col+1))
        if (i % chnum == 0):
            plt.legend(loc=2, ncol=1)
            plt.savefig('result/' + 'var_' + testnum + '_ch' + str(i//chnum) + '.png')
            plt.clf()
        i += 1
        
### File traversal ###
def fileTraversal():
    start = time.time()
    path = 'C:/Users/rss75/Desktop/IMS Dataset/data'
    i = 0
    for testnum in os.listdir(path):
        dim = getData(path + '/' + testnum + '/' + os.listdir(path+"/"+testnum+"/")[0]).shape[0]
        dstime = time.time()
        featuredata = np.array([]).reshape(0, dim)
        for filename in os.listdir(path+"/"+testnum+"/")[0:]:
            buf = getData(path+"/"+testnum+"/"+filename).reshape(1, dim)
            featuredata = np.append(featuredata, buf, axis=0)
        savePlot(featuredata,testnum)
        np.savetxt('result/' + 'skew_' + testnum + '.txt', featuredata, delimiter=',', fmt='%.5f')
        print('Processed ' + str(featuredata.shape[0]) + ' files in ' + str(timedelta(seconds=time.time()-dstime)))
    print('--- Time completed: ' + str(timedelta(seconds=time.time()-start)))

### Statistical Extraction ###
def getStats(mat):
    #k = getKurtosis(mat)
    s = getSkewness(mat)
    #v = getVariance(mat)
    #e = getEntropy(mat)
    #m = getMean(mat)
    return s

# Kurtosis
def getKurtosis(mat):
    kur = np.array([])
    for idx in range(0, mat.shape[1]):
        k = stats.kurtosis(mat[idx, :], fisher=False)
        kur = np.append(kur, np.array([k]), axis=0)
    return kur

# Mean
def getMean(mat):
    mean = np.array([])
    for idx in range(0, mat.shape[1]):
        m = stats.trim_mean(mat[idx, :],0.05)
        mean = np.append(mean, np.array([m]), axis=0)
    return mean

# Skewness
def getSkewness(mat):
    skew = np.array([])
    for idx in range(0, mat.shape[1]):
        s = stats.skew(mat[idx, :], axis=0, bias=True)
        skew = np.append(skew, np.array([s]), axis=0)
    return skew

# Variance
def getVariance(mat):
    var = np.array([])
    for idx in range(0, mat.shape[1]):
        v = np.var(mat[idx, :])
        var = np.append(var, np.array([v]), axis=0)
    return var

# Entropy
def getEntropy(mat):
    ent = np.array([])
    for idx in range(0, mat.shape[1]):
        e = stats.entropy(mat[idx, :])
        ent = np.append(ent, np.array([e]), axis=0)
    return ent

### Get Probability Distribution ###
def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(stats, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

# Retrieve data from file
def getDataFile(filename):
    filearr = np.genfromtxt(filename, delimiter=",", dtype='float')
    return filearr

# Exponential function
def curve_func(x,a,b,c,d,e,f,g,h):
    #return a*(np.exp(x*b)**2) + c*(np.exp(x*b)) + d
    #return a*(np.exp(x*b)) + c
    return a*(x**7) + b*(x**6) + c*(x**5) + d*(x**4) + e*(x**3) + f*(x**2) + g*x + h

def curve_func2(x,a,b,c):
    #return a*(np.exp(x*b)**2) + c*(np.exp(x*b)) + d
    return a*(np.exp(x*b)) + c
    #return a*(x**3) + b*(x**2) + c*(x**1) + d

# Plot data from file
def plotDataFitting(data):
    n = data.shape[0]
    order = 5
    #print(n)
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    chnum = data.shape[1]//4
    i = 1
    xdata = np.linspace(1, data.shape[0]+1, n)
    for col in range(0, data.shape[1]):
        print("Processing column " + str(col) + "...")
        #print(data[:,col].shape)
        #print(range(1,data[:,col].shape[0]))
        #p = np.poly1d(np.polyfit(xdata,data[:,col],order))
        popt_exp, pcov_exp = curve_fit(curve_func, xdata, data[:,col], p0=[10, -10, 10, 3, 6, 4, 2, 3], maxfev=300)
        print(popt_exp)
        plt.plot(xdata, data[:,col], '.', color="black", label='ch'+str(col+1))
        #plt.plot(xdata, p(xdata), '-', color=palette(col), label='ch'+str(col+1))
        plt.plot(xdata, curve_func(xdata,*popt_exp))
        plt.ylim(min(data[:,col])-0.15*(max(data[:,col])-min(data[:,col])),max(data[:,col])+0.4*(max(data[:,col])-min(data[:,col])))
        # xdata, data[:, col], '.', 
        if (i % chnum == 0):
            plt.legend(loc=2, ncol=1)
            #plt.savefig('result/' + 'datafit_ch' + str(i//chnum) + '_ord' + str(order) + '.png',dpi=1500)
            plt.show()
            plt.clf()
        i += 1

# Plot data from file (cumulative sum of all channel)
def plotDataFittingCumulative(data):
    n = data.shape[0]
    order = 5
    #print(n)
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    chnum = data.shape[1]//8
    i = 1
    xdata = np.linspace(1, data.shape[0]+1, n)
    ycuml = np.zeros(data.shape[0])
    for col in range(0, data.shape[1]):
        print("Processing column " + str(col) + "...")
        #print(data[:,col].shape)
        #print(range(1,data[:,col].shape[0]))
        #p = np.poly1d(np.polyfit(xdata,data[:,col],order))
        popt_exp, pcov_exp = curve_fit(curve_func, xdata, data[:,col], p0=[0.0001, -10, 10, 3, 7, 2, 3, 1], maxfev=200)
        #popt_exp, pcov_exp = curve_fit(curve_func2, xdata, data[:,col], p0=[5, 0.3, 10], maxfev=1500)
        for i in range(data.shape[0]):
            ycuml[i] += curve_func(xdata[i],*popt_exp)
            #ycuml[i] += curve_func2(xdata[i],*popt_exp)
    #plt.plot(xdata, data[:,col], '.', color="black", label='ch'+str(col+1))
    #plt.plot(xdata, p(xdata), '-', color=palette(col), label='ch'+str(col+1))
    plt.title('Variance data trend over time (cumulative)')
    plt.xlabel('Time step')
    plt.ylabel('Variance')
    plt.plot(xdata, ycuml)
    plt.ylim(min(ycuml)-0.15*(max(ycuml)-min(ycuml)),max(ycuml)+0.4*(max(ycuml)-min(ycuml)))
    # xdata, data[:, col], '.', 
    
    #plt.show()
    plt.savefig('result/var_1st_plot_noscat.png',dpi=1500)
    plt.clf()

# Curve fitting (scipy.optimize)
def getCurveFitting(data):
    data_fitted = np.array([]).reshape(0,1)
    xdata = np.linspace(1, data.shape[0], data.shape[0])
    print(xdata)
    print(data)
    for column in range(0, data.shape[1]):
        #p = np.poly1d(np.polyfit(xdata,data[:,column],3))
        popt_exp, pcov_exp = scipy.optimize.curve_fit(exponential, xdata, data[:,column], p0=[1,-0.5, 1])
        print(popt_exp)
        #data_fitted = np.append(data_fitted, np.array([p(0),p(1),p(2),p(3)]).reshape(1,4), axis=0)
        data_fitted = np.append(data_fitted, np.array(popt_exp), axis=0)
        print(data_fitted)
    return data_fitted

# Long Short Term Memory Neural-Networks

### Main ###
fileTraversal()
#print(getDataFile('result/var_1st_test.txt'))
#plotData(getDataFile('result/var_1st_test.txt'))
#plotDataFittingCumulative(getDataFile('result/var_1st_test.txt'))
#plotDataFitting(getDataFile('result/kurt_2nd_test.txt'))