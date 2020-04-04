import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np



data = pd.read_csv('arrhythmia.csv',header=None)
print(data.shape)

count=0
for i in range(0,452):
    for j in range(0,280):
        if (data.iloc[i,j]=='?'):
            count =count+1
print(count)
data = data.replace('?', np.NaN)

data.drop(columns = 13, inplace=True)
print(data.shape)

imputer = KNNImputer(n_neighbors=5, weights="uniform")
data_no_missing = imputer.fit_transform(data)
data_no_missing=pd.DataFrame(data_no_missing)