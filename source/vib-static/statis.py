import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

month = 7
day = 1
hour = 0
minute = 0
headers = ['time', 'x', 'y', 'z']

def RMScalc(df,c):
    return np.sqrt(df[c].pow(2).sum()/len(df[c]))
def Clearance(df,c):
    root = (df[c].abs().pow(0.5).sum()/1000)**2
    return root

Feats = pd.DataFrame(columns=['Skx','Sky','RMSx','RMSy','Kx','Ky','CFx','CFy','P2Px','P2Py','filename'])
listDay = []
listJam = []
listMin = []

while day < 19:
  
  #HARI
  hari = [month, day]
  for i in range(0,len(hari)):
    if hari[i] < 10:
      hari[i] = '0'+str(hari[i])
    else:
      hari[i] = str(hari[i])
  
  #bikin list hari
  fileDay = "accel_cont_20_{d[0]}_{d[1]}*".format(d = hari)
  listDay = !find /content/drive/My\ Drive/Tugas\ Akhir/data/accel-vi-data-cont/accel/ -maxdepth 1 -name {fileDay}
  listDay.sort()
  #print('panjang listDay awal: ', len(listDay))
  
  if hour == 24:
    hour = 0

  #JAM
  while hour <24:
    #file terakhir
    if (day == 18 and hour >= 13):
      break
    listJam = []
    if minute >= 60:
      minute = 0
    jam = [month,day,hour]
    for i in range(0,len(jam)):
      if jam[i] < 10:
        jam[i] = '0'+str(jam[i])
      else:
        jam[i] = str(jam[i])
    
    #bikin list jam
    fileJam = "20_{d[0]}_{d[1]}_{d[2]}".format(d = jam)
    listJam = [s for s in listDay if fileJam in s]
    listJam.sort()
    listDay.sort()
    listDay = listDay[len(listJam):]
    #print('panjang listDay akhir: ', len(listDay))

    #MENIT
    while minute <60:
    #file terakhir
      if (day == 18 and hour == 12 and minute >= 53):
        break
      listMin=[]
      menit = [month,day,hour, minute]
      for i in range(0,len(menit)):
        if menit[i] < 10:
          menit[i] = '0'+str(menit[i])
        else:
          menit[i] = str(menit[i])

      #bikin list menit
      fileMin = "20_{d[0]}_{d[1]}_{d[2]}_{d[3]}".format(d = menit)
      listMin = [s for s in listJam if fileMin in s]

      if len(listMin) != 0:

        listMin.sort()
        listJam.sort()
        listJam = listJam[len(listMin):]

        #ngebaca file menit
        for e in range(0,len(listMin)):
          if e == 0:
            dfRaw = pd.read_csv(listMin[e], names = headers)
          else:
            dfRaw = dfRaw.append(pd.read_csv(listMin[e], names = headers))
          
        SKx = dfRaw['x'].skew()
        SKy = dfRaw['y'].skew()
        RMSx = RMScalc(dfRaw,'x')
        RMSy = RMScalc(dfRaw,'y')
        Kx = dfRaw['x'].kurt()
        Ky = dfRaw['y'].kurt()
        CFx = float(dfRaw['x'].max()/RMSx)
        CFy = float(dfRaw['y'].max()/RMSy)
        P2Px = dfRaw['x'].max() - dfRaw['x'].min()
        P2Py = dfRaw['y'].max() - dfRaw['y'].min()
        filename = listMin[e][-27:-5]
        #filename = '{}-{}-{} {}:{}'.format(20,month,day,hour,minute)
        list = [SKx,SKy,RMSx,RMSy,Kx,Ky,CFx,CFy,P2Px,P2Py,filename]

        Feats.loc[idx] = list
        idx = idx +1
        
        if idx % 500 == 0 and len(listMin)>0:
          Feats.to_csv('/content/drive/My Drive/Tugas Akhir/{}{}.csv'.format(idx,listMin[e][-27:-5]))
          print('abis ngolah waktu : ',day, " ",hour," ", minute, " ",listMin[e])
        
      minute = minute + 1

    hour = hour +1
    
  day = day +1

Feats.to_csv('/content/drive/My Drive/Tugas Akhir/{}_{}_{}_{}_{}.csv'.format(idx,20,str(month),str(day),str(hour),str(minute)))
