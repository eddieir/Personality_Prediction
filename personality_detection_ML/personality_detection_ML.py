'''

The database we are working with classifies people into 16 distinct personality types showing their last 50 tweets, separated by "|||".

Our goal will be to create new columns based on the content of the tweets, in order to create a predictive model. As we will see, this can be quite tricky and our creativity comes into play when analysing the content of the tweets.

We begin by importing our dataset and showing some info, for an initial exploratory analysis.

'''

import re
import numpy as np
import collections
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(
    '/home/eddie/Desktop/python/personality_detection/mbti-type/mbti_1.csv')
print(df.head(10))
print("*"*40)
print(df.info())

df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)
print(df.head())

'''

Exploratory data analysis

We may use it for one reason or for another, but one thing we can do is printing a violin plot.

At the end I did not use it at all, but it is always nice to have the ability do some visual analysis for further investigations.
'''
plt.figure(figsize=(15, 10))
sns.violinplot(x='type', y='words_per_comment',
               data=df, inner=None, color='lightgray')
sns.stripplot(x='type', y='words_per_comment', data=df, size=4, jitter=True)
plt.ylabel('Words per comment')
plt.savefig('word_per_comment_VS_type.png')


'''
There's quite a lot of information there.

Creating new columns showing the amount of questionmarks per comment, exclamations or other types will be useful later on, as we will see. This are the examples I came up with, but here is where creativity comes into play.

We can also perform joint plots, pair plots and heat maps to explore relationship between data, just for fun.
'''
df['http_per_comment'] = df['posts'].apply(lambda x: x.count('http')/50)
df['music_per_comment'] = df['posts'].apply(lambda x: x.count('music')/50)
df['question_per_comment'] = df['posts'].apply(lambda x: x.count('?')/50)
df['img_per_comment'] = df['posts'].apply(lambda x: x.count('jpg')/50)
df['excl_per_comment'] = df['posts'].apply(lambda x: x.count('!')/50)
df['ellipsis_per_comment'] = df['posts'].apply(lambda x: x.count('...')/50)

plt.figure(figsize=(15, 10))
sns.jointplot(x='words_per_comment',
              y='ellipsis_per_comment', data=df, kind='kde')
plt.savefig('ellipsisPerComment_VS_wordsPerComment.png')


'''
So it seems there's a large correlation between words per comment ant the ellipsis the user types per comment!
'''
i = df['type'].unique()
k = 0
for m in range(0, 2):
    for n in range(0, 6):
        df_2 = df[df['type'] == i[k]]
        sns.jointplot(x='words_per_comment',
                      y='ellipsis_per_comment', data=df_2, kind="hex")
        plt.title(i[k])
        k += 1
        # plt.show()

i = df['type'].unique()
k = 0
TypeArray = []
PearArray = []
for m in range(0, 2):
    for n in range(0, 6):
        df_2 = df[df['type'] == i[k]]
        pearsoncoef1 = np.corrcoef(
            x=df_2['words_per_comment'], y=df_2['ellipsis_per_comment'])
        pear = pearsoncoef1[1][0]
        print(pear)
        TypeArray.append(i[k])
        PearArray.append(pear)
        k += 1
plt.show()

TypeArray = [x for _, x in sorted(zip(PearArray, TypeArray))]
PearArray = sorted(PearArray, reverse=True)
print(PearArray)
print(TypeArray)
plt.scatter(TypeArray, PearArray)
plt.savefig('TypwArray_VS_PearArray.png')


'''
Data preprocessing
To get a further insight on our dataset, we can first create 4 new columns dividing the people by introversion/extroversion, intuition/sensing, and so on.

When it comes to performing machine learning, trying to distinguish between two categories is much easier than distinguishing between 16 categories. 
We will check that later on.Dividing the data in 4 small groups will perhaps be more useful when it comes to accuracy.

'''
map1 = {"I": 0, "E": 1}
map2 = {"N": 0, "S": 1}
map3 = {"T": 0, "F": 1}
map4 = {"J": 0, "P": 1}
df['I-E'] = df['type'].astype(str).str[0]
df['I-E'] = df['I-E'].map(map1)
df['N-S'] = df['type'].astype(str).str[1]
df['N-S'] = df['N-S'].map(map2)
df['T-F'] = df['type'].astype(str).str[2]
df['T-F'] = df['T-F'].map(map3)
df['J-P'] = df['type'].astype(str).str[3]
df['J-P'] = df['J-P'].map(map4)
print(df.head(10))
