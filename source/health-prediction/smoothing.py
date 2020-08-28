import pandas as pd
import matplotlib.pyplot as plt

trainData = pd.read_csv('source/health-prediction/traindata.csv')
trainDataEWM = trainData.ewm(alpha=0.05).mean()
trainData['HIs'] = trainDataEWM['HI'].to_numpy()
print(trainData.head())
print(trainData.size)

xData = list(range(len(trainData['HI'])))
yData = trainData['HI']
yData2 = trainData['HIs']

plt.plot(xData, yData, 'r--', linewidth=0.5)
plt.plot(xData, yData2, 'k-', linewidth=2)
plt.show()


""" a = [1,2,3,2,3,4,3,4,5,2,5,4,5,6,5,6,7,6,7,7,6,7,7,8,7,7,8,9,7,8,9]
d = {
    'd1': a
}
df = pd.DataFrame(data=d)
df_ewm = df.ewm(alpha=0.2).mean()
df.plot()
df_ewm.plot() """