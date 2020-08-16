import matplotlib.pyplot as plt
import numpy as np
import random

a = [a for a in range(21)]
b = [a[i]*random.randint(1,4) for i in range(21)]
c = [a[i]*random.randint(1,4) for i in range(21)]

fig = plt.figure(figsize=(8,6), dpi=100)
plt.get_current_fig_manager().window.state('zoomed')

ax1 = fig.add_subplot(111, frame_on=True)
ax2 = fig.add_subplot(111, frame_on=False)

p1, = ax1.plot(a,b, 'k--', linewidth=0.5)
ax1.set_xticks(list(range(0,21,2)))
ax1.set_xlim([0,20])
ax1.set_ylim([0,100])
ax1.grid(True)

p2, = ax2.plot(a,c, 'k-', linewidth=2)
ax2.set_xticks(list(range(0,21,2)))
ax2.set_xlim([0,20])
ax2.set_ylim([0,100])
ax2.grid(True)

ax1.legend(
    (p1,p2),
    ('Plot 1', 'Plot 2'),
    loc='upper right',
    fontsize=16
)

plt.show()