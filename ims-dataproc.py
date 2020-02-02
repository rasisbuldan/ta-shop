### To do ###
# 1. Convert filename to date
# 2. Entropy calculate probability distribution values


### Library ###
import numpy as np                  # Array and matrix operation
import os                           # File operation
import scipy.stats as stats         # Statistical library
from io import StringIO             # String handling
import matplotlib.pyplot as plt     # Plot operation
import time                         # Time
from datetime import timedelta      # Time formatting
#import scipy.io                # Matlab file handling

### Global Variable ###
buf2 = np.empty([1, 8])
buf1 = np.array([])

### File extraction ###
def getData(filename):
    print("Processing " + filename)
    filearr = np.genfromtxt(filename, delimiter="\t", dtype='float')
    for idx in range(0,filearr.shape[1]):
        print("--------------------- idx: " + str(idx))
        get_best_distribution(filearr[idx,:])
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
            plt.savefig('result/' + 'ent_' + testnum + '_ch' + str(i//chnum) + '.png')
            plt.clf()
        i += 1
        
### File traversal ###
def fileTraversal():
    start = time.time()
    path = 'C:/Users/Stoorm/Desktop/IMS/data'
    i = 0
    for testnum in os.listdir(path):
        dim = getData(path + '/' + testnum + '/' + os.listdir(path+"/"+testnum+"/")[0]).shape[0]
        dstime = time.time()
        featuredata = np.array([]).reshape(0, dim)
        for filename in os.listdir(path+"/"+testnum+"/")[0:]:
            buf = getData(path+"/"+testnum+"/"+filename).reshape(1, dim)
            featuredata = np.append(featuredata, buf, axis=0)
        savePlot(featuredata,testnum)
        np.savetxt('result/' + 'ent_' + testnum + '.txt', featuredata, delimiter=',', fmt='%.5f')
        print('Processed ' + str(featuredata.shape[0]) + ' files in ' + str(timedelta(seconds=time.time()-dstime)))
    print('--- Time completed: ' + str(timedelta(seconds=time.time()-start)))

### Statistical Extraction ###
def getStats(mat):
    k = getKurtosis(mat)
    #s = getSkewness(mat)
    #v = getVariance(mat)
    #e = getEntropy(mat)
    return k

# Kurtosis
def getKurtosis(mat):
    kur = np.array([])
    for idx in range(0, mat.shape[1]):
        k = stats.kurtosis(mat[idx, :], fisher=False)
        kur = np.append(kur, np.array([k]), axis=0)
    return kur

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
        skew = np.append(var, v, axis=0)
    return var

# Entropy
def getEntropy(mat):
    ent = np.array([])
    for idx in range(0, mat.shape[1]):
        e = stats.entropy(mat[idx, :])
        ent = np.append(ent, np.array([e]), axis=0)
    print(ent)
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

### Loop ###
fileTraversal()
