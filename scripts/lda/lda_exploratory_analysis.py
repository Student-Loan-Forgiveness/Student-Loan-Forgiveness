'''
LDA - Exploratory Analysis
'''

# import specific functions
from exploratory_functions import *
from vectorizing_functions import *


## IMPORT DATA ##
# import newsapi data
newsapi_labeled = pd.read_csv('../send_files/cleaning/newsapi_data_cleaned/newsapi_labeled_2_14_25.csv')

# import reddit data
reddit_student_loan_forgiveness = pd.read_csv('../send_files/cleaning/reddit_data_cleaned/labeled_student_loan_forgiveness.csv')
reddit_student_loans = pd.read_csv('../send_files/cleaning/reddit_data_cleaned/labeled_student_loans.csv')

# add labels for search
reddit_student_loan_forgiveness['search'] = 'reddit_student_loan_forgiveness'
reddit_student_loans['search'] = 'reddit_student_loans'

# concatenate
reddit_labeled = pd.concat([reddit_student_loan_forgiveness, reddit_student_loans], ignore_index=True)


## PREPARE DATA ##
# reduce to labeled specific
newsapi_labeled.dropna(subset='Bias Specific', inplace=True)

# reset index
newsapi_labeled.reset_index(drop=True, inplace=True)

# apply function - further specific cleaning
newsapi_labeled['cleaned_article'] = newsapi_labeled['article'].apply(specific_cleaning)

# apply function - lemmatize
newsapi_labeled['lemmatized_article'] = newsapi_labeled['cleaned_article'].apply(lemmatize_article)

# remove additional words
newsapi_labeled['lemmatized_article'] = newsapi_labeled['lemmatized_article'].apply(lambda x: remove_additional_words(x, ['wa', 'ha', 'tt', 've', 'wt', 'tn']))

# text data
newsapi_text_data = newsapi_labeled['lemmatized_article'].tolist()

# apply function - further specific cleaning
reddit_labeled['cleaned_content'] = reddit_labeled['author_content_aggregated'].apply(specific_cleaning)

# apply function - lemmatize
reddit_labeled['lemmatized_content'] = reddit_labeled['cleaned_content'].apply(lemmatize_article)

# remove additional words
reddit_labeled['lemmatized_content'] = reddit_labeled['lemmatized_content'].apply(lambda x: remove_additional_words(x, ['wa', 'ha', 'tt', 've', 'wt', 'tn']))

# get sentence length
reddit_labeled['word_count'] = reddit_labeled['lemmatized_content'].apply(lambda x: len(x.split()))

# also removes reactions and single word replies - since the aim isn't to track 
reddit_labeled = reddit_labeled[reddit_labeled['word_count'] >= 15]

# reset index
reddit_labeled.reset_index(drop=True, inplace=True)

# aggregate into author schema
retain_columns = ['author', 'lemmatized_content']
reddit_author_schema = aggregate_into_schema(reddit_labeled, retain_columns, 'author')

# reddit text data
reddit_text_data = reddit_author_schema['lemmatized_content'].tolist()

## COUNTVECTORIZER ##
# newsapi
params_max = {'stop_words': 'english'}
cv_max_newsapi = vectorize_to_df(newsapi_text_data, input_type='content', vectorizer_type='count', params=params_max)

# 10% maximum features
params_tenth = {'stop_words': 'english',
                'max_features': int(cv_max_newsapi.shape[1] / 10)}
cv_tenth_newsapi = vectorize_to_df(newsapi_text_data, input_type='content', vectorizer_type='count', params=params_tenth)

# save snippet
cv_tenth_newsapi.head(10).to_csv('data_prep/cv_tenth_newsapi.csv', index=False)

# reddit
params_max = {'stop_words': 'english'}
cv_max_reddit = vectorize_to_df(reddit_text_data, input_type='content', vectorizer_type='count', params=params_max)

# 10% maximum features
params_tenth = {'stop_words': 'english',
                'max_features': int(cv_max_reddit.shape[1] / 10)}
cv_tenth_reddit = vectorize_to_df(reddit_text_data, input_type='content', vectorizer_type='count', params=params_tenth)

# save snippet
cv_tenth_reddit.head(10).to_csv('data_prep/cv_tenth_reddit.csv', index=False)

## PERFORM LDA ##
# newsapi - model
lda_model_newsapi, lda_object_newsapi = run_lda(cv_tenth_newsapi, 3, iterations=100, learning='online')

# newsapi - words
create_lda_top_list(lda_model_newsapi, lda_object_newsapi, num_words=15, fontsize_base=115, save_path='lda_words_newsapi.png')

# newsapi - plots
create_lda_frequency_plot(lda_model_newsapi, num_words=15, save_path='lda_plot_newsapi.png', subplot_dims=[3, 1], max_limits=True, plot_size=(12, 8), text_size=12)

# reddit - model
lda_model_reddit, lda_object_reddit = run_lda(cv_tenth_reddit, 3, iterations=100, learning='online')

# reddit - words
create_lda_top_list(lda_model_reddit, lda_object_reddit, num_words=15, fontsize_base=115, save_path='lda_words_reddit.png')

# reddit - plots
create_lda_frequency_plot(lda_model_reddit, num_words=15, save_path='lda_plot_reddit.png', subplot_dims=[3, 1], max_limits=True, plot_size=(12, 8), text_size=12)
