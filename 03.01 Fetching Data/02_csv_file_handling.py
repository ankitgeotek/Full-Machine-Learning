import pandas as pd

'Opening a csv file from Local machine'
df=pd.read_csv(r'A:\Work Docs\Data Analyst work\Campus X\03 Machine Learning\Project\aug_train.csv')


'Opening a csv file from a url'

import requests
from io import StringIO

url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0"}
req = requests.get(url, headers=headers)
data = StringIO(req.text)

df_url=pd.read_csv(data)
df_url.head()


#Seperation Parameter changing
# tab in place of comma     sep='\t'
pd.read_csv(r'A:\Work Docs\Data Analyst work\Campus X\03 Machine Learning\Project\movie_titles_metadata.tsv',sep='\t')


#Giving name to the column
pd.read_csv(r'A:\Work Docs\Data Analyst work\Campus X\03 Machine Learning\Project\movie_titles_metadata.tsv',sep='\t',names=['S.no','Title','ReleaseYear','Rating','Votes','Genres'])


#Index column
df=pd.read_csv(r'03 Machine Learning\Project\aug_train.csv',index_col='enrollee_id')
df

#Header Parameter
pd.read_csv(r'03 Machine Learning\Project\test.csv')
pd.read_csv(r'03 Machine Learning\Project\test.csv',header=1)

#use_cols Parameter --Load only perticular columns
pd.read_csv(r'03 Machine Learning\Project\test.csv',header=1,usecols=['gender','city'])


#Squeeze Parameter- To load only one column and treat it as a series
pd.read_csv(r'03 Machine Learning\Project\test.csv',header=1,usecols=['gender'],squeeze=True)
 

#skip rows/nrows Parameter
pd.read_csv(r'03 Machine Learning\Project\test.csv',skiprows=[2,3])

#nrows- to ristrict no of rows to be loaded

pd.read_csv(r'03 Machine Learning\Project\IPL Matches 2008-2020.csv',nrows=100)


#Encoding Parameter
pd.read_csv(r'03 Machine Learning\Project\zomato.csv')

pd.read_csv(r'03 Machine Learning\Project\zomato.csv',encoding='latin-1')

#skipping Bad Lines
pd.read_csv(r'03 Machine Learning\Project\test.csv',error_bad_lines=False)

#dtype
pd.read_csv(r"03 Machine Learning\Project\aug_train.csv",dtype={'target':int})

#Handling Data -string to date type
pd.read_csv(r"03 Machine Learning\Project\IPL Matches 2008-2020.csv",parse_dates=['date']).info()

#convertors
def rename(name):
    if name=='Royal Challenger Bangalore':
        return 'RCB'
    else:
        name
rename("Royal Challenger Bangalore")


pd.read_csv(r"03 Machine Learning\Project\IPL Matches 2008-2020.csv",converters={'team1':rename})['team1']

#na_values parameter
pd.read_csv(r'03 Machine Learning\Project\aug_train.csv',na_values=['NaN','-'])

#Loading data in chunks

dfs=pd.read_csv(r'03 Machine Learning\Project\aug_train.csv',chunksize=5000)

for chunks in dfs:
    print(chunks.shape)
    



