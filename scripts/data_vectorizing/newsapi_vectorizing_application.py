'''
NewsAPI Vectorizing Application
'''

# import libraries
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import os

# import specific functions
from vectorizing_functions import *

# import data
newsapi_labeled = pd.read_csv('../cleaning/newsapi_data_cleaned/newsapi_labeled_2_14_25.csv')

# labels
newsapi_labels = ['source', 'author', 'Bias Specific', 'Bias Numeric', 'date', 'search']

# apply function - further specific cleaning
newsapi_labeled['cleaned_article'] = newsapi_labeled['article'].apply(specific_cleaning)

# apply function - lemmatize
newsapi_labeled['lemmatized_article'] = newsapi_labeled['cleaned_article'].apply(lemmatize_article)

# apply function - stemmaize
newsapi_labeled['stemmatized_article'] = newsapi_labeled['cleaned_article'].apply(stem_article)

# parameters for version 1
params_1 = {'stop_words': 'english',
            'max_features': 200}

# content for overall
text_data_1 = newsapi_labeled['cleaned_article'].tolist()

# content for overall - lemmatized
text_data_2 = newsapi_labeled['lemmatized_article'].tolist()

# content for overall - stemmed
text_data_3 = newsapi_labeled['stemmatized_article'].tolist()

'''
Part 1a: CountVectorizer - Overall
'''
# vectorizing overall - countvectorizer
cv_1 = vectorize_to_df(text_data_1, input_type='content', vectorizer_type='count', params=params_1)
cv_1_labeled = pd.concat([newsapi_labeled[newsapi_labels], cv_1], axis=1)
cv_1_labeled.to_csv('newsapi_vectorized/newsapi_overall_cv.csv', index=False)

# wordcloud overall - countvectorizer
cv_1_wc = create_word_cloud(cv_1.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(cv_1_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('newsapi_vectorized/newsapi_overall_cv.png', dpi=500)
plt.show()

# top frequency barplot - countvectorizer
cv_1_top = cv_1.sum(axis=0).reset_index()
cv_1_top.columns = ['word', 'count']
cv_1_top = cv_1_top.nlargest(10, 'count')

# plot
plt.figure(figsize=(12, 8))
sns.barplot(cv_1_top, x='count', y='word')
plt.xlabel('Counts')
plt.ylabel('Words')
plt.title('Most Frequent Words - Overall')
plt.savefig('newsapi_vectorized/most_frequent_words_overall.png', dpi=300)
plt.show()

'''
Part 1b: TfidfVectorizer - Overall
'''
# vectorizing overall - tfidftvectorizer
tf_1 = vectorize_to_df(text_data_1, input_type='content', vectorizer_type='tfid', params=params_1)
tf_1_labeled = pd.concat([newsapi_labeled[newsapi_labels], tf_1], axis=1)
tf_1_labeled.to_csv('newsapi_vectorized/newsapi_overall_tf.csv', index=False)

'''
Part 2a: CountVectorizer - Overall - Lemmatized
'''
# vectorizing overall - countvectorizer
cv_2 = vectorize_to_df(text_data_2, input_type='content', vectorizer_type='count', params=params_1)
cv_2_labeled = pd.concat([newsapi_labeled[newsapi_labels], cv_2], axis=1)
cv_2_labeled.to_csv('newsapi_vectorized/newsapi_overall_cv_lemmatized.csv', index=False)

# wordcloud overall - countvectorizer
cv_2_wc = create_word_cloud(cv_2.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(cv_2_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('newsapi_vectorized/newsapi_overall_cv_lemmatized.png', dpi=500)
plt.show()

# top frequency barplot - countvectorizer
cv_2_top = cv_2.sum(axis=0).reset_index()
cv_2_top.columns = ['word', 'count']
cv_2_top = cv_2_top.nlargest(10, 'count')

# plot
plt.figure(figsize=(12, 8))
sns.barplot(cv_2_top, x='count', y='word')
plt.xlabel('Counts')
plt.ylabel('Words')
plt.title('Most Frequent Words - Overall - Lemmatized')
plt.savefig('newsapi_vectorized/most_frequent_words_overall_lemmatized.png', dpi=300)
plt.show()

'''
Part 2b: TfidfVectorizer - Overall Lemmatized
'''
# vectorizing overall - tfidftvectorizer
tf_2 = vectorize_to_df(text_data_2, input_type='content', vectorizer_type='tfid', params=params_1)
tf_2_labeled = pd.concat([newsapi_labeled[newsapi_labels], tf_2], axis=1)
tf_2_labeled.to_csv('newsapi_vectorized/newsapi_overall_tf_lemmatized.csv', index=False)

'''
Part 3a: CountVectorizer - Overall - Stemmatized
'''
# vectorizing overall - countvectorizer
cv_3 = vectorize_to_df(text_data_3, input_type='content', vectorizer_type='count', params=params_1)
cv_3_labeled = pd.concat([newsapi_labeled[newsapi_labels], cv_3], axis=1)
cv_3_labeled.to_csv('newsapi_vectorized/newsapi_overall_cv_stemmatized.csv', index=False)

# wordcloud overall - countvectorizer
cv_3_wc = create_word_cloud(cv_3.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(cv_3_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('newsapi_vectorized/newsapi_overall_cv_stemmatized.png', dpi=500)
plt.show()

# top frequency barplot - countvectorizer
cv_3_top = cv_3.sum(axis=0).reset_index()
cv_3_top.columns = ['word', 'count']
cv_3_top = cv_3_top.nlargest(10, 'count')

# plot
plt.figure(figsize=(12, 8))
sns.barplot(cv_3_top, x='count', y='word')
plt.xlabel('Counts')
plt.ylabel('Words')
plt.title('Most Frequent Words - Overall - Stemmatized')
plt.savefig('newsapi_vectorized/most_frequent_words_overall_stemmatized.png', dpi=300)
plt.show()

'''
Part 3b: TfidfVectorizer - Overall Stemmatized
'''
# vectorizing overall - tfidftvectorizer
tf_3 = vectorize_to_df(text_data_3, input_type='content', vectorizer_type='tfid', params=params_1)
tf_3_labeled = pd.concat([newsapi_labeled[newsapi_labels], tf_3], axis=1)
tf_3_labeled.to_csv('newsapi_vectorized/newsapi_overall_tf_stemmatized.csv', index=False)

'''
Part 4: Lemmatized Labels

- Lemmatizing seems to aggregate the text data and retains meaning in words.
- There are two methods to examine the labels:
    - vectorize over entire dataset and then subset on the biases
    - subset first on the biases and then vectorize  
- By subsetting second, the maximum word count is still reflective of the corpus overall,
- although the counts and frequencies may differ
- By subsetting first, the maximum word count is more reflective of the differences between the labeled corpuses,
therefore this analysis will subset first
'''

# subset
left_subset = newsapi_labeled[newsapi_labeled['Bias Specific']=='Left'].reset_index(drop=True)
lean_left_subset = newsapi_labeled[newsapi_labeled['Bias Specific']=='Lean Left'].reset_index(drop=True)
center_subset = newsapi_labeled[newsapi_labeled['Bias Specific']=='Center'].reset_index(drop=True)
lean_right_subset = newsapi_labeled[newsapi_labeled['Bias Specific']=='Lean Right'].reset_index(drop=True)
right_subset = newsapi_labeled[newsapi_labeled['Bias Specific']=='Right'].reset_index(drop=True)

# text data
left_text_data = left_subset['lemmatized_article'].tolist()
lean_left_text_data = lean_left_subset['lemmatized_article'].tolist()
center_text_data = center_subset['lemmatized_article'].tolist()
lean_right_text_data = lean_right_subset['lemmatized_article'].tolist()
right_text_data = right_subset['lemmatized_article'].tolist()

# vectorizing
left_vectorized = vectorize_to_df(left_text_data, input_type='content', vectorizer_type='count', params=params_1)
lean_left_vectorized = vectorize_to_df(lean_left_text_data, input_type='content', vectorizer_type='count', params=params_1)
center_vectorized = vectorize_to_df(center_text_data, input_type='content', vectorizer_type='count', params=params_1)
lean_right_vectorized = vectorize_to_df(lean_right_text_data, input_type='content', vectorizer_type='count', params=params_1)
right_vectorized = vectorize_to_df(right_text_data, input_type='content', vectorizer_type='count', params=params_1)

# apply labels
left_vectorized_labeled = pd.concat([left_subset[newsapi_labels], left_vectorized], axis=1)
lean_left_vectorized_labeled = pd.concat([lean_left_subset[newsapi_labels], lean_left_vectorized], axis=1)
center_vectorized_labeled = pd.concat([center_subset[newsapi_labels], center_vectorized], axis=1)
lean_right_vectorized_labeled = pd.concat([lean_right_subset[newsapi_labels], lean_right_vectorized], axis=1)
right_vectorized_labeled = pd.concat([right_subset[newsapi_labels], right_vectorized], axis=1)

# save dataframes
left_vectorized_labeled.to_csv('newsapi_vectorized/left_cv.csv', index=False)
lean_left_vectorized_labeled.to_csv('newsapi_vectorized/lean_left_cv.csv', index=False)
center_vectorized_labeled.to_csv('newsapi_vectorized/center_cv.csv', index=False)
lean_right_vectorized_labeled.to_csv('newsapi_vectorized/lean_right_cv.csv', index=False)
right_vectorized_labeled.to_csv('newsapi_vectorized/right_cv.csv', index=False)

# wordcloud - left
left_wc = create_word_cloud(left_vectorized.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(cv_2_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('newsapi_vectorized/newsapi_left.png', dpi=500)
plt.show()

# wordcloud - lean left
lean_left_wc = create_word_cloud(lean_left_vectorized.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(lean_left_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('newsapi_vectorized/newsapi_lean_left.png', dpi=500)
plt.show()

# wordcloud - center
center_wc = create_word_cloud(center_vectorized.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(center_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('newsapi_vectorized/newsapi_center.png', dpi=500)
plt.show()

# wordcloud - lean right
lean_right_wc = create_word_cloud(lean_right_vectorized.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(lean_right_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('newsapi_vectorized/newsapi_lean_right.png', dpi=500)
plt.show()

# wordcloud - right
right_wc = create_word_cloud(right_vectorized.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(right_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('newsapi_vectorized/newsapi_right.png', dpi=500)
plt.show()

# top frequency barplot - left
left_top = left_vectorized.sum(axis=0).reset_index()
left_top.columns = ['word', 'count']
left_top = left_top.nlargest(10, 'count')

# plot
plt.figure(figsize=(12, 8))
sns.barplot(left_top, x='count', y='word')
plt.xlabel('Counts')
plt.ylabel('Words')
plt.title('Most Frequent Words - Left')
plt.savefig('newsapi_vectorized/most_frequent_words_left.png', dpi=300)
plt.show()

# top frequency barplot - lean left
lean_left_top = lean_left_vectorized.sum(axis=0).reset_index()
lean_left_top.columns = ['word', 'count']
lean_left_top = lean_left_top.nlargest(10, 'count')

# plot
plt.figure(figsize=(12, 8))
sns.barplot(lean_left_top, x='count', y='word')
plt.xlabel('Counts')
plt.ylabel('Words')
plt.title('Most Frequent Words - Lean Left')
plt.savefig('newsapi_vectorized/most_frequent_words_lean_left.png', dpi=300)
plt.show()

# top frequency barplot - center
center_top = center_vectorized.sum(axis=0).reset_index()
center_top.columns = ['word', 'count']
center_top = center_top.nlargest(10, 'count')

# plot
plt.figure(figsize=(12, 8))
sns.barplot(center_top, x='count', y='word')
plt.xlabel('Counts')
plt.ylabel('Words')
plt.title('Most Frequent Words - Center')
plt.savefig('newsapi_vectorized/most_frequent_words_center.png', dpi=300)
plt.show()

# top frequency barplot - lean right
lean_right_top = lean_right_vectorized.sum(axis=0).reset_index()
lean_right_top.columns = ['word', 'count']
lean_right_top = lean_right_top.nlargest(10, 'count')

# plot
plt.figure(figsize=(12, 8))
sns.barplot(lean_right_top, x='count', y='word')
plt.xlabel('Counts')
plt.ylabel('Words')
plt.title('Most Frequent Words - Lean Right')
plt.savefig('newsapi_vectorized/most_frequent_words_lean_right.png', dpi=300)
plt.show()

# top frequency barplot - right
right_top = right_vectorized.sum(axis=0).reset_index()
right_top.columns = ['word', 'count']
right_top = right_top.nlargest(10, 'count')

# plot
plt.figure(figsize=(12, 8))
sns.barplot(right_top, x='count', y='word')
plt.xlabel('Counts')
plt.ylabel('Words')
plt.title('Most Frequent Words - Right')
plt.savefig('newsapi_vectorized/most_frequent_words_right.png', dpi=300)
plt.show()

'''
Snippets Section
'''
# import data - overall
newsapi_overall_cv = pd.read_csv('newsapi_vectorized/newsapi_overall_cv.csv')
newsapi_overall_tf = pd.read_csv('newsapi_vectorized/newsapi_overall_tf.csv')
newsapi_ovalll_cv_lemmatizd = pd.read_csv('newsapi_vectorized/newsapi_overall_cv_lemmatized.csv')
newsapi_ovalll_tf_lemmatizd = pd.read_csv('newsapi_vectorized/newsapi_overall_tf_lemmatized.csv')
newsapi_ovalll_cv_stemmatized = pd.read_csv('newsapi_vectorized/newsapi_overall_cv_stemmatized.csv')
newsapi_ovalll_tf_stemmatized = pd.read_csv('newsapi_vectorized/newsapi_overall_tf_stemmatized.csv')

# import data - political biases
newsapi_left = pd.read_csv('newsapi_vectorized/left_cv.csv')
newsapi_lean_left = pd.read_csv('newsapi_vectorized/lean_left_cv.csv')
newsapi_center = pd.read_csv('newsapi_vectorized/center_cv.csv')
newsapi_lean_right = pd.read_csv('newsapi_vectorized/lean_right_cv.csv')
newsapi_right = pd.read_csv('newsapi_vectorized/right_cv.csv')

# snippets - overall
newsapi_overall_cv.head(10).to_csv('newsapi_vectorized/snippets/newsapi_overall_cv.csv', index=False)
newsapi_overall_tf.head(10).to_csv('newsapi_vectorized/snippets/newsapi_overall_tf.csv', index=False)
newsapi_ovalll_cv_lemmatizd.head(10).to_csv('newsapi_vectorized/snippets/newsapi_overall_cv_lemmatized.csv', index=False)
newsapi_ovalll_tf_lemmatizd.head(10).to_csv('newsapi_vectorized/snippets/newsapi_overall_tf_lemmatized.csv', index=False)
newsapi_ovalll_cv_stemmatized.head(10).to_csv('newsapi_vectorized/snippets/newsapi_overall_cv_stemmatized.csv', index=False)
newsapi_ovalll_tf_stemmatized.head(10).to_csv('newsapi_vectorized/snippets/newsapi_overall_tf_stemmatized.csv', index=False)

# snippets - political biases
newsapi_left.head(10).to_csv('newsapi_vectorized/snippets/left.csv', index=False)
newsapi_lean_left.head(10).to_csv('newsapi_vectorized/snippets/lean_left.csv', index=False)
newsapi_center.head(10).to_csv('newsapi_vectorized/snippets/center.csv', index=False)
newsapi_lean_right.head(10).to_csv('newsapi_vectorized/snippets/lean_right.csv', index=False)
newsapi_right.head(10).to_csv('newsapi_vectorized/snippets/right.csv', index=False)
