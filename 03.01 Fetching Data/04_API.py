'''

https://api.themoviedb.org/3/movie/top_rated?api_key=17ff9f4bf16b1c672e9e339da6176f1e

id
title
release_date
overview
popularity
vote_average
vote_count
8851*7

'''

import pandas as pd
import requests     #to create http request
response=requests.get('https://api.themoviedb.org/3/movie/top_rated?api_key=17ff9f4bf16b1c672e9e339da6176f1e')
#<Response [200]> means everything went   well
response.json()#to convert incoming data in json formate
response.json()['results']

df=pd.DataFrame(response.json()['results'])[['id','title','overview','popularity','vote_average','vote_count']]
df.info()
df.shape
df.head()

#getting all pages through loop
df=pd.DataFrame()

for i in range(1,429):
    response = requests.get('https://api.themoviedb.org/3/movie/top_rated?api_key=17ff9f4bf16b1c672e9e339da6176f1e&language=en-US&page={}'.format(i))
    temp_df = pd.DataFrame(response.json()['results'])[['id','title','overview','release_date','popularity','vote_average','vote_count']]
    df = df.append(temp_df,ignore_index=True)






























