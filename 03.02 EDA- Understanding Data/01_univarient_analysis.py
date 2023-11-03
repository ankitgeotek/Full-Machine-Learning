import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r'A:\Work Docs\DATA Set\titanic_train.csv')
df.head()
df.info()


df['Survived']=df['Survived'].astype('category')


sns.countplot(x=df["Pclass"])
sns.countplot(x=df['Survived'],orient='h')
plt.show()

df['Survived'].value_counts().plot(kind='bar')
plt.show()

sns.countplot(x=df['Pclass'])
plt.show()

sns.countplot(x=df['Sex'])
plt.show()





'''Using PieChart for %'''
df['Pclass'].value_counts().plot(kind='pie',autopct='%.2f')
plt.show()

df['Sex'].value_counts().plot(kind='pie',autopct='%.2f')
plt.show()

df['Pclass'].value_counts().plot(kind='pie',autopct='%.2f')
plt.show()

sns.countplot(x=df['Sex'])
plt.show()


#Numerical Data
#Histogram
plt.hist(df['Age'],bins=50)
plt.show()

plt.hist(df['Age'],bins=4)
plt.show()

#Distplot- Distribution plot
sns.distplot(x=df['Age'])
plt.show()

#box Plot
sns.boxplot(x=df['Age'])
plt.show()

sns.boxplot(x=df['Fare'])
plt.show()


#Skewness
df['Age'].skew()


'''
Bivariant and Multi varient Analysis
'''

tips=sns.load_dataset('tips')
flights=sns.load_dataset('flights')
iris=sns.load_dataset('iris')

tips
flights
iris

#scatter plot(Numerical-Numerical)
sns.scatterplot(x=tips['total_bill'],y=tips['tip'])
plt.show()

sns.scatterplot(x=tips['total_bill'],y=tips['tip'],hue=tips['sex'],style=tips['smoker'],size=tips['size'])
plt.show()


titanic=pd.read_csv(r'A:\Work Docs\DATA Set\titanic_train.csv')
titanic

#Barplot

sns.barplot(x=titanic['Pclass'],y=titanic['Age'])
plt.show()


sns.barplot(x=titanic['Pclass'],y=titanic['Fare'],hue=titanic['Sex'])
plt.show()

sns.barplot(x=titanic['Pclass'],y=titanic['Age'],hue=titanic['Sex'])
plt.show()

#distplot
sns.displot(x=titanic['Age'])
plt.show()

sns.distplot(x=titanic[titanic['Survived']==0]['Age'],hist=False)
sns.distplot(x=titanic[titanic['Survived']==1]['Age'],hist=False)
plt.show()

#Cross tab and heatmap
pd.crosstab(titanic['Pclass'],titanic['Survived'])

sns.heatmap(pd.crosstab(titanic['Pclass'],titanic['Survived']))
plt.show()

titanic.groupby('Pclass').mean()['Survived']
titanic.groupby('Pclass').mean()

'Cluster Map -Categorical vs Categorical'
pd.crosstab(titanic['SibSp'],titanic['Survived'])

sns.clustermap(pd.crosstab(titanic['SibSp'],titanic['Survived']))
plt.show()

sns.clustermap(pd.crosstab(titanic['Parch'],titanic['Survived']))
plt.show()


#Pairplot- Multiple Numerical column-Automaticallt detact all the numerical data column

sns.pairplot(iris)
plt.show()

'Using categorical column as hue in Pairplot'
sns.pairplot(iris,hue='species')
plt.show()
iris

#Line plot (Numerical-Numerical) prefered with time data
flights.head()
flights.groupby('year')

