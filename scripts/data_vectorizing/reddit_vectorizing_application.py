'''
Reddit Vectorizing Application
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
reddit_1 = pd.read_csv('../cleaning/reddit_data_cleaned/labeled_student_loan_forgiveness.csv')
reddit_2 = pd.read_csv('../cleaning/reddit_data_cleaned/labeled_student_loans.csv')

# create column for search parameters
reddit_1['search'] = 'student_loan_forgiveness'
reddit_2['search'] = 'student_loans'

# concatenate
reddit = pd.concat([reddit_1, reddit_2], ignore_index=True)

# double check for duplicates
reddit.drop_duplicates(subset=['url', 'author'], inplace=True)

'''
Aggregation Schemas:
    - thread - author (INITIAL FORMAT): corpus where each file is an author's aggregated text within a unique thread
    - subreddit - author: corpus where each file is an author's aggregated text within a unique subreddit
    - thread: corpus where each file is overall aggregated text within a unique thread (author's combined)
    - subreddit: corpus where each file is overall aggregated text within a subreddit (threads combined)
    - author: corpus where each file is overall aggregated text by authors across every thread

Strategy:
    - perform general preprocessing
    - lemmatize
    - analyze across the above schemas
'''

'''
Part 1: Aggregation Schemas
'''
# remove AutoModerator
thread_author = reddit.copy()
thread_author = thread_author[thread_author['author'] != 'AutoModerator'].reset_index(drop=True)

# preprocessing
thread_author['content_cleaned'] = thread_author['author_content_aggregated'].apply(specific_cleaning)

# lemmatization
thread_author['schema_1'] = thread_author['content_cleaned'].apply(lemmatize_article)

# aggregation schema 1 (thread-author)
text_thread_author = thread_author['schema_1'].tolist()

# thread-author labels
thread_author_labels = ['author', 'url', 'subreddit', 'search']

# aggregation schema 2 (subreddit-author)
subreddit_author = thread_author.groupby(['author', 'subreddit'])['schema_1'].apply(lambda content: ' '.join(content)).reset_index()
text_subreddit_author = subreddit_author['schema_1'].tolist()

# subreddit-author labels
subreddit_author_labels = ['author', 'subreddit']

# aggregation schema 3 (threads)
threads = thread_author.groupby(['url'])['schema_1'].apply(lambda content: ' '.join(content)).reset_index()
text_threads = threads['schema_1'].tolist()

# threads labels
threads_labels = ['url']

# aggregation schema 4 (subreddits)
subreddits = thread_author.groupby(['subreddit'])['schema_1'].apply(lambda content: ' '.join(content)).reset_index()
text_subreddits = subreddits['schema_1'].tolist()

# subreddits labels
subreddits_labels = ['subreddit']

# aggregation schema 5 (authors)
authors = thread_author.groupby(['author'])['schema_1'].apply(lambda content: ' '.join(content)).reset_index()
text_authors = authors['schema_1'].tolist()

# authors labels
authors_labels = ['author']

# vectorizer parameters
schema_params = {'stop_words': 'english',
                 'max_features': 200}

'''
Part 2: Thread-Author Schema
'''
# aggregation schema 1
cv_1 = vectorize_to_df(text_thread_author, input_type='content', vectorizer_type='count', params=schema_params)
cv_1_labeled = pd.concat([thread_author[thread_author_labels], cv_1], axis=1)
cv_1_labeled.to_csv('reddit_vectorized/thread_author.csv', index=False)

# wordcloud overall - countvectorizer
cv_1_wc = create_word_cloud(cv_1.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(cv_1_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('reddit_vectorized/wc_thread_author.png', dpi=500)
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
plt.title('Most Frequent Words - Authors by Thread')
plt.savefig('reddit_vectorized/most_frequent_words_thread_author.png', dpi=300)
plt.show()

'''
Part 3: Subreddit-Author Schema
'''
# aggregation schema 2
cv_2 = vectorize_to_df(text_subreddit_author, input_type='content', vectorizer_type='count', params=schema_params)
cv_2_labeled = pd.concat([subreddit_author[subreddit_author_labels], cv_2], axis=1)
cv_2_labeled.to_csv('reddit_vectorized/subreddit_author.csv', index=False)

# wordcloud overall - countvectorizer
cv_2_wc = create_word_cloud(cv_2.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(cv_2_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('reddit_vectorized/wc_subreddit_author.png', dpi=500)
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
plt.title('Most Frequent Words - Authors by Subreddit')
plt.savefig('reddit_vectorized/most_frequent_words_subreddit_author.png', dpi=300)
plt.show()

'''
Part 4: Thread Schema
'''
# aggregation schema 3
cv_3 = vectorize_to_df(text_threads, input_type='content', vectorizer_type='count', params=schema_params)
cv_3_labeled = pd.concat([threads[threads_labels], cv_3], axis=1)
cv_3_labeled.to_csv('reddit_vectorized/threads.csv', index=False)

# wordcloud overall - countvectorizer
cv_3_wc = create_word_cloud(cv_3.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(cv_3_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('reddit_vectorized/wc_threads.png', dpi=500)
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
plt.title('Most Frequent Words - Threads')
plt.savefig('reddit_vectorized/most_frequent_words_threads.png', dpi=300)
plt.show()

'''
Part 5: Subreddit Schema
'''
# aggregation schema 4
cv_4 = vectorize_to_df(text_subreddits, input_type='content', vectorizer_type='count', params=schema_params)
cv_4_labeled = pd.concat([subreddits[subreddits_labels], cv_4], axis=1)
cv_4_labeled.to_csv('reddit_vectorized/subreddits.csv', index=False)

# wordcloud overall - countvectorizer
cv_4_wc = create_word_cloud(cv_4.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(cv_4_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('reddit_vectorized/wc_subreddits.png', dpi=500)
plt.show()

# top frequency barplot - countvectorizer
cv_4_top = cv_4.sum(axis=0).reset_index()
cv_4_top.columns = ['word', 'count']
cv_4_top = cv_4_top.nlargest(10, 'count')

# plot
plt.figure(figsize=(12, 8))
sns.barplot(cv_4_top, x='count', y='word')
plt.xlabel('Counts')
plt.ylabel('Words')
plt.title('Most Frequent Words - Subreddits')
plt.savefig('reddit_vectorized/most_frequent_words_subreddits.png', dpi=300)
plt.show()

'''
Part 6: Author Schema
'''
# aggregation schema 5
cv_5 = vectorize_to_df(text_authors, input_type='content', vectorizer_type='count', params=schema_params)
cv_5_labeled = pd.concat([authors[authors_labels], cv_5], axis=1)
cv_5_labeled.to_csv('reddit_vectorized/authors.csv', index=False)

# wordcloud overall - countvectorizer
cv_5_wc = create_word_cloud(cv_5.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(cv_5_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('reddit_vectorized/wc_authors.png', dpi=500)
plt.show()

# top frequency barplot - countvectorizer
cv_5_top = cv_5.sum(axis=0).reset_index()
cv_5_top.columns = ['word', 'count']
cv_5_top = cv_5_top.nlargest(10, 'count')

# plot
plt.figure(figsize=(12, 8))
sns.barplot(cv_5_top, x='count', y='word')
plt.xlabel('Counts')
plt.ylabel('Words')
plt.title('Most Frequent Words - Authors')
plt.savefig('reddit_vectorized/most_frequent_words_authors.png', dpi=300)
plt.show()

'''
One Additional CountVectorizer for max_df and min_df using the first schema

- min_df: 0.25
- max_df: 0.75
'''
# additional parameters
additional_params = {'stop_words': 'english',
                     'min_df': 0.1,
                     'max_df': 0.9}

# additional parameters cv
cv_add = vectorize_to_df(text_thread_author, input_type='content', vectorizer_type='count', params=additional_params)
cv_add_labeled = pd.concat([thread_author[thread_author_labels], cv_add], axis=1)
cv_add_labeled.to_csv('reddit_vectorized/thread_author_additional.csv', index=False)

# wordcloud overall - countvectorizer
cv_add_wc = create_word_cloud(cv_add.sum(axis=0).to_dict(), cloud_method='frequency')
plt.figure(figsize=(12, 8))
plt.imshow(cv_add_wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('reddit_vectorized/wc_thread_author_additional.png', dpi=500)
plt.show()

# top frequency barplot - countvectorizer
cv_add_top = cv_add.sum(axis=0).reset_index()
cv_add_top.columns = ['word', 'count']
cv_add_top = cv_add_top.nlargest(10, 'count')

# plot
plt.figure(figsize=(12, 8))
sns.barplot(cv_add_top, x='count', y='word')
plt.xlabel('Counts')
plt.ylabel('Words')
plt.title('Most Frequent Words - Authors by Thread Additional')
plt.savefig('reddit_vectorized/most_frequent_words_thread_author_additional.png', dpi=300)
plt.show()


'''
Snippets Section
'''
# schema 1: thread-author
cv_1_labeled.head(10).to_csv('reddit_vectorized/snippets/thread_author.csv', index=False)

# schema 2: subreddit-author
cv_2_labeled.head(10).to_csv('reddit_vectorized/snippets/subreddit_author.csv', index=False)

# schema 3: threads
cv_3_labeled.head(10).to_csv('reddit_vectorized/snippets/threads.csv', index=False)

# schema 4: subreddits
cv_4_labeled.head(10).to_csv('reddit_vectorized/snippets/subreddits.csv', index=False)

# schema 5: authors
cv_5_labeled.head(10).to_csv('reddit_vectorized/snippets/authors.csv', index=False)

# additional parameters
cv_add_labeled.head(10).to_csv('reddit_vectorized/snippets/thread_author_additional.csv', index=False)
