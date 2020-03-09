import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

news_label = pd.read_csv('/Users/mac/Downloads/news.csv')
print(news_label.shape)
print(news_label.head(5))
labels = news_label.label
print(labels.head())

x_train, x_test, y_train, y_test = train_test_split(news_label['word'], labels, test_size=0.2, random_state = 7)


#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train.values.astype(str)) 
tfidf_test=tfidf_vectorizer.transform(x_test.values.astype(str))

pac= PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print('accuracy is:{}%'.format(round(score*100,2)))



