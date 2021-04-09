import pandas as pd
import os
# from IPython.display import display

currentDirectory = os.getcwd()
final_train = pd.read_csv(currentDirectory+"/final_train.csv") 

final_train = final_train.drop(['Unnamed: 0'], axis = 1)

cols = ['Activity', 'subject']
filled = final_train.copy()
filled[filled.columns.difference(cols)] = final_train.groupby(cols).transform(lambda x: x.fillna(x.mean()))[filled.columns.difference(cols)]
print(filled.head())
print(filled.isnull().sum().sum())

filled.to_csv(currentDirectory+'/data_nona.csv', index=False)