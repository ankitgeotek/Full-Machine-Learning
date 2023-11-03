"A:\Work Docs\DATA Set\Social_Network_Ads.csv"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"A:\Work Docs\DATA Set\Social_Network_Ads.csv")
df.sample(5)

df=df.iloc[:,2:]
df.sample(5)

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(df.drop('Purchased',axis=1),df['Purchased'],test_size=0.3,random_state=0)

x_train.shape, x_test.shape
y_train.shape, y_test.shape

#Standard Scalar class in sklearn
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

#fit the scaler to the train set, it will learn the Parameter
scaler.fit(x_train)
scaler.mean_

#transform the train test sets
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)

type(x_train_scaled)    #it is a numpy array


x_train_scaled=pd.DataFrame(x_train_scaled,columns=x_train.columns)
x_test_scaled=pd.DataFrame(x_test_scaled,columns=x_test.columns)

np.round(x_train.describe(),1)
np.round(x_train_scaled.describe(),1)


#Observing effect of scaling through graphs
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.scatter(x_train['Age'], x_train['EstimatedSalary'])
ax1.set_title("Before Scaling")
ax2.scatter(x_train_scaled['Age'], x_train_scaled['EstimatedSalary'],color='red')
ax2.set_title("After Scaling")
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(x_train['Age'], ax=ax1)
sns.kdeplot(x_train['EstimatedSalary'], ax=ax1)

# after scaling
ax2.set_title('After Standard Scaling')
sns.kdeplot(x_train_scaled['Age'], ax=ax2)
sns.kdeplot(x_train_scaled['EstimatedSalary'], ax=ax2)
plt.show()


'Age-Comparing distribution shape before and after scaling'
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Age Distribution Before Scaling')
sns.kdeplot(x_train['Age'], ax=ax1)

# after scaling
ax2.set_title('Age Distribution After Standard Scaling')
sns.kdeplot(x_train_scaled['Age'], ax=ax2)
plt.show()


'Salary-Comparing distribution shape before and after scaling'
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Salary Distribution Before Scaling')
sns.kdeplot(x_train['EstimatedSalary'], ax=ax1)

# after scaling
ax2.set_title('Salary Distribution Standard Scaling')
sns.kdeplot(x_train_scaled['EstimatedSalary'], ax=ax2)
plt.show()


#Why scaling is important?
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr_scaled=LogisticRegression()

lr.fit(x_train,y_train)
lr_scaled.fit(x_train_scaled,y_train)


y_pred=lr.predict(x_test)
y_pred_scaled=lr.predict(x_test_scaled)

y_pred = lr.predict(x_test)
y_pred_scaled = lr_scaled.predict(x_test_scaled)
y_test.shape
y_test
y_pred
y_pred_scaled


from sklearn.metrics import accuracy_score
print("Actual",accuracy_score(y_test,y_pred))
print("Scaled",accuracy_score(y_test,y_pred_scaled))


accuracy_score(y_test,y_pred)
print(f'actual accuracy score :{accuracy_score(y_test,y_pred)}')
print(f'Scaled accuracy score :{accuracy_score(y_test,y_pred_scaled)}')
























































































































