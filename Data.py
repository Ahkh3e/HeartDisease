import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')

print("num of rows", train.shape[0])
print("num of rows", train.shape[1])
print("\n")
train.info()
print("\n")
print("All data values available")


plt.subplot2grid((2,3),(0,0))
train.DEATH_EVENT.value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
plt.title ("Death event")


plt.subplot2grid((2,3),(0,1))
train.sex[train.DEATH_EVENT == 1 ].value_counts(normalize = True).plot(kind = "bar", alpha = 0.5,color = ['b','r'])
plt.title ("Sex and death")


plt.subplot2grid((2,3),(1,0))
train.DEATH_EVENT[train.sex == 0 ].value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
plt.title ("Men Death event")

plt.subplot2grid((2,3),(1,1))
train.DEATH_EVENT[train.sex == 1 ].value_counts(normalize = True).plot(kind = "bar", alpha = 0.5, color = 'r')
plt.title ("Women Death event")

plt.show()
