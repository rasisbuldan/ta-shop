import numpy as np
import matplotlib.pyplot as plt

a = np.array([
    [1,0],
    [1,1],
    [1,0.3],
    [1,0.4],
    [1,0.7],
    [5,0],
    [5,0.1],
    [5,0.4],
    [5,0.8],
    [5,0.2],
    [5,0.2],
    [5,1],
    [7,1],
    [7,0],
    [7,0.2],
    [7,0.3],
    [7,0.4],
    [7,0.7]
])

#b = np.split(a,(np.where(np.diff(a[:,0]))[0]+1))
#print(b.shape)

fig, ax1 = plt.subplots()
ax1.scatter(a[:,0], a[:,1])
#ax1.set_xticklabels(['a','b','c'])