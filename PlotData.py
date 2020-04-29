import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
file = input("Enter company name\n")
data = pd.read_csv("data/"+file+".csv")
#print(data)
sns.relplot(data = data, x='Date', y='Adj Close', kind='line')
plt.show()

