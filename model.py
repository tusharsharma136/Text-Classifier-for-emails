import numpy as np 
import pandas as pd 
import plotly 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"
import os
import regex
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import nltk
nltk.download('omw-1.4')
from wordcloud import WordCloud
from nltk.corpus import stopwords, words
from nltk.tokenize import WordPunctTokenizer, word_tokenize
from string import punctuation
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import pickle
import streamlit as st
df=pd.read_csv("combine.csv")
# Visualisation of counts of text files from different categories
data_file=df
df1=df
category_count=data_file["Category"].value_counts()
category_count=dict(category_count)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import re

en_stopwords = set(stopwords.words('english'))
word_lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove non-alphabetic characters, extra whitespace, and newlines
    text= str(text);
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    words = [w for w in words if not w in en_stopwords]

    # Lemmatize the words
    words = [word_lemmatizer.lemmatize(w) for w in words]

    # Join the words back into a string
    cleaned_text = ' '.join(words)
    return cleaned_text

# Tokenizing
def tokenize_content(content):
    return clean_text(content)

df['cleaned_content'] = df['Content'].apply(tokenize_content)

df.style.hide_index()

# Joining the text files of one particular category for word cloud further

def join_content(l):
 ans=''
 for i in l:
   ans = ans+ ' '+ i
 return ans
crime_text = df[df['Category'] == 'Crime']['cleaned_content'].values
crime_text = join_content(crime_text)

entertainment_text = df[df['Category'] == 'Entertainment']['cleaned_content'].values
entertainment_text = join_content(entertainment_text)

politics_text = df[df['Category'] == 'Politics']['cleaned_content'].values
politics_text = join_content(politics_text)

science_text = df[df['Category'] == 'Science']['cleaned_content'].values

science_text = join_content(science_text)

# Creating the world clouds 
def wordcloud(content):
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='black',
                    min_font_size = 10).generate(content)
    return wordcloud

from wordcloud import WordCloud, STOPWORDS

#Defined a class called wordcloud1 to get a information about
#frequency of each word in the texts of each category
def wordcloud1(content):
    wordcloud1 = WordCloud(width = 800, height = 800,
                    background_color ='black',
                    stopwords = STOPWORDS,
                    min_font_size = 10)
    wordcloud1.generate_from_text(content)
    
    # Get word frequencies as a dictionary
    word_freq = wordcloud1.process_text(content)
    # Sort words by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words
 

# Dropping the Duplicates
import matplotlib.pyplot as plt
df.drop_duplicates(subset=['ID', 'Content'], inplace=True, keep=False)
after_count=df['Category'].value_counts()

# Balancing the dataset

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


category_counts = {}

# Loop over each row and update the count for the corresponding category
for index, row in df.iterrows():
    category = row['Category']
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

# Find the minimum count value from the dictionary
min_count = min(category_counts.values())

balanced_df = pd.DataFrame(columns=df.columns)

# Loop over each category and select the first n rows
for category in df['Category'].unique():
    category_rows = df.loc[df['Category'] == category][:min_count]
    balanced_df = pd.concat([balanced_df, category_rows])

# Shuffle the rows of the new DataFrame
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# Split the new DataFrame into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(balanced_df['cleaned_content'], balanced_df['Category'], test_size=0.2, random_state=42)

# Transform the text data into numerical features using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)
pickle.dump(clf,open('log_model.pkl','wb'))
import joblib
joblib.dump(vectorizer, 'vectorizer.pkl')
