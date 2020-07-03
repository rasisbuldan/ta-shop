import sys
from datetime import datetime, timedelta
import time

t0 = datetime.now()
print(t0)
time.sleep(2)
t1 = datetime.now()
print(t1)
td = t1 - t0
tdop = timedelta(hours=1)
print(td)
print(td.seconds)
print((t1 + tdop).strftime('%H:%M:%S'))