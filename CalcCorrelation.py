import numpy as np
import pandas as pd

base = input("Enter the primary stock\n")
stlst = input("Enter one or more stock separated by space to be compared with\n").split(" ")

def calcReturns(arr):
	res = np.subtract(np.log(arr[:-1]),np.log(arr[1:]))
	return res

primary = pd.read_csv("data/"+base+".csv")
primary = calcReturns(primary['Adj Close'].to_numpy())

for item in stlst:
	compare = pd.read_csv("data/"+item+".csv")
	compare = calcReturns(compare['Adj Close'].to_numpy())
	r = np.corrcoef(primary,compare)
	print(r)	


