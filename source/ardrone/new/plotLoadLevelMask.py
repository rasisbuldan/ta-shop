import pandas as pd
import matplotlib.pyplot as plt

# Load csv
cd = pd.read_csv('comparedataHI.csv').iloc[10:]
cd['rHIsss'] = cd.ewm(alpha=0.5).mean()['rHIss']

fig = plt.figure()
ax1 = fig.add_subplot(111, frame_on=True)
ax2 = fig.add_subplot(111, frame_on=False)

p1, = ax1.plot(cd['loadNum'], 'r-')
ax1.yaxis.tick_right()
ax1.set_ylim([0,7])
ax1.set_yticks([0,1,2,3,4,5,6])
ax1.set_xlabel('Waktu (s)')
ax1.set_ylabel('Indikator Kesehatan')

p2 = ax2.plot(cd['rHIsss'], 'k-')
ax2.set_ylim([0,1])
ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

ax1.grid(True)
plt.show()