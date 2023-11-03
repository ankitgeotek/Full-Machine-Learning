import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"A:\Work Docs\DATA Set\wine_data.csv",header=None,usecols=[0,1,2])
df
df.columns=['Class Label','Alcohal','Malic Acid']
df


sns.kdeplot(x=df['Alcohal'])
plt.show()

color_dict={1:'red',2:'blue',3:'green'}
sns.scatterplot(x=df['Alcohal'],y=df['Malic Acid'],hue=df['Class Label'],palette=color_dict)
plt.show()


'''Splitting data for train test'''
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(df.drop('Class Label',axis=1),df['Class Label'],test_size=0.3,random_state=0)

x_train,x_test
x_train.shape,x_test.shape

y_train,y_test
y_train.shape, y_test.shape


'''Importing MaxMinScaler for Normalising the data'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#Fit the scaler to train set, it will Learn the parameter
scaler.fit(x_train) #it will calculate max and min  for preprosessing

x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)

#when we use sklearn library it convert our data to numpy array 
#to further operate on data we have to convert it back to pandas dataframe
x_train_scaled=pd.DataFrame(x_train_scaled,columns=x_train.columns)
x_test_scaled=pd.DataFrame(x_test_scaled,columns=x_test.columns)

type(x_train_scaled)
type(x_train)

np.round(x_train.describe(),1)
np.round(x_train_scaled.describe(),1)

#scatter plot
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.scatter(x_train['Alcohal'], x_train['Malic Acid'],c=y_train)
ax1.set_title("Before Scaling")
ax2.scatter(x_train_scaled['Alcohal'], x_train_scaled['Malic Acid'],c=y_train)
ax2.set_title("After Scaling")
plt.show()

#distribution plots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(x_train['Alcohal'], ax=ax1)
sns.kdeplot(x_train['Malic Acid'], ax=ax1)

# after scaling
ax2.set_title('After Standard Scaling')
sns.kdeplot(x_train_scaled['Alcohal'], ax=ax2)
sns.kdeplot(x_train_scaled['Malic Acid'], ax=ax2)
plt.show()














































