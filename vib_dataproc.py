import numpy as np
import matplotlib.pyplot as plt

### File extraction ###
def getData(filename):
    print("Processing " + filename)
    filearr = np.genfromtxt(filename, delimiter=",", dtype='float')
    print(filearr.shape)
    return filearr

vib_data = getData("vibrasi2.csv")
time = vib_data[:,0]
x = vib_data[:,1]
y = vib_data[:,2]
z = vib_data[:,3]

plt.title("Vibration x")
plt.plot(time[0:150],x[0:150],'-')
plt.show()
plt.clf()

plt.title("Vibration y")
plt.plot(time[0:150],y[0:150],'-')
plt.show()
plt.clf()

plt.title("Vibration z")
plt.plot(time[0:150],z[0:150],'-')
plt.show()
plt.clf()