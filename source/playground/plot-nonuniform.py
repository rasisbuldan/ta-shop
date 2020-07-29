import matplotlib.pyplot as plt
import numpy as np

""" a = [1,2,3,6,7,10,15]
b = [3,6,4,9,1,4,10]

plt.plot(a,b)
plt.xlim(0,20)
plt.xticks(np.arange(0, 21, 2))
plt.grid(True)
plt.show() """

import matplotlib.pyplot as plt

x_values1=[100,220,270,320,450]
y_values1=[10,20,39,40,50]

x_values2=[150,200,250,300,350]
y_values2=[10,20,30,40,50]

x_values3=[1,2,3,4,5]
y_values3=[1,2,2,4,1]


fig=plt.figure()
ax=fig.add_subplot(211, label="1")
ax2=fig.add_subplot(211, label="2", frame_on=False)
ax3=fig.add_subplot(212, label="3")

p1, = ax.plot(x_values1, y_values1, color="C0", label='plot A')
#ax.legend(loc='upper right')
#ax.set_xlabel("x label 1", color="C0")

ax.set_ylabel("y label 1", color="C0")
ax.tick_params(axis='x', colors="C0")
ax.tick_params(axis='y', colors="C0")
ax.set_xlim([0,500])
ax.set_xticks([])

p2, = ax2.plot(x_values2, y_values2, color="C1", label='plot B')
#ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
#ax2.legend(loc='upper right')
#ax2.set_xlabel('x label 2', color="C1") 
ax2.set_ylabel('y label 2', color="C1")       
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors="C1")
ax2.tick_params(axis='y', colors="C1")
ax2.set_ylim([0,100])
ax2.set_xlim([0,500])
ax2.set_xticks(np.arange(0,501,50))
#plt.legend([p1,p2],['plot 1', 'plot 2'])


ax3.plot(x_values3, y_values3, color="C3")
ax3.set_xticks([])
ax3.set_yticks([])

ax.legend((p1,p2), ('pA','pB'), loc='upper right')
plt.grid(True)
plt.show()