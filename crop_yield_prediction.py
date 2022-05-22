# -*- coding: utf-8 -*-
"""
Created on Sun May  8 14:00:07 2022

@author: ballanicherry984
"""


import pandas
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
#______________________________Data selection___________________________
data = pandas.read_csv("crop_production.csv")
print("----------------------------------------------------------")
print("Data Selection")
print()
print(data.head(10))
print()
print("-----------------------------------------------------------")
#_______________________________Preprocessing____________________________
df = data.copy()
print("Dropping null values")
df.dropna(axis=0, inplace=True)
print("Checking the data")
print(df.isna().sum())
#df["Crop"].value_counts()
#_________________________________________________________________
print("Taking only the crops which have data more than 1500")
crop_count = df["Crop"].value_counts()
df = df.loc[df["Crop"].isin(crop_count.index[crop_count > 1500])]
print(df.head(10))
print()
print()
#__________________________________________________________________
print("printing the crop names whose count is greater than 1500")
names=list(set(df["Crop"].values))
print(names)
print()
print()
#_____________________Giving the crop name as input_________________
cro=input("Enter the crop name:")
Name = df[(df["Crop"] == cro)]
print("dispyaling the selected crop data")
print(Name.head(10))
print()
print()
#________________________________________________________________________
dt = Name.copy()
le = LabelEncoder()
scaler = MinMaxScaler()
dt["district"] = le.fit_transform(dt["District_Name"])
dt['season'] = le.fit_transform(dt["Season"])
#dt["area"] = scaler.fit_transform(dt[["Area"]])
dt["state"] = le.fit_transform(dt["State_Name"])
print("Data after preprocessing")
print(dt.head(10))
print()
print()
#________________________Data splitting__________________________________
X = dt[["Area", "district", "season", "state"]]
y = dt["Production"]
split=int(len(dt)*0.85)
#k=dt["Area"].values
#k=k[split:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
print()
print()
print("printing the training data")
print(X_train.head())
print()
print()
print(y_train.head())
print()
print()
print("printing the testing data")
print(X_test.head())
print()
print()
print(y_test.head())
print()
print()
#_____________________________Random Forest______________________________________
model = RandomForestRegressor()
model.fit(X_train, y_train)
print("Score of Random forest regressor:",model.score(X_test, y_test))
print("Prediting the production by taking the test data")
y_predict=model.predict(X_test)
print()
print()
print("predicted values",y_predict[:10])
print()
print()
print("Actual values",y_test[:10])
print()
print()
'''#_____________________________Decision Tree Regressor______________________
regressor = DecisionTreeRegressor(random_state = 0) 
regressor.fit(X_train, y_train)
y_pre=regressor.predict(X_test)
print("Score of Decision Tree Regressor:",regressor.score(X_test,y_test))
print()
print()
'''

#____________________plotting the actual production with area_______________
plt.figure(figsize=(14, 10))
sns.regplot(Name["Area"], Name["Production"]).set(title='Area vs Actual production')
plt.show()

#___________________plotting the predicted production with area_____________
h1=X_test['Area'].values
h1=h1.reshape((-1,1))
plt.figure(figsize=(14, 10))
sns.regplot(h1,y_predict).set(title='Area vs Predicted production')
plt.show()





