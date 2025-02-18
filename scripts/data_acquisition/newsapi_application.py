'''
NEWSAPI Extraction and Subsequent Website Scraping
'''

# import libraries
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import datetime
import os
from dotenv import load_dotenv

# import specific functions
from newsapi_functions import *

# newsapi secrets
load_dotenv()
api_key = os.getenv('NEWSAPI_KEY')

'''
Initial Data Retrieval
'''

# extract articles with topic 'Student Loan Forgiveness'
newsapi_extraction = extract_newsapi('student loan forgiveness', api_key, sort_by='popularity', iterate=True)

# reformat extracted article information
news_df = format_newsapi(newsapi_extraction)

# get unique sources
unique_sources = get_unique_sources(news_df)

# search for unreviewed articles
unreviewed_sources = check_sources(news_df)

# scrape and concatenate
articles_df = scrape_and_concatenate(news_df)

# save dataframes
news_df.to_csv('newsapi_data/extraction_1_18_25.csv', index=False)
articles_df.to_csv('newsapi_data/scraped_1_18_25.csv', index=False)

'''
Additional Data Retrievals - Round 1

- query: 'student loan forgiveness'
'''

# load data
extracted_data = pd.read_csv('newsapi_data/extraction_1_18_25.csv')
scraped_data = pd.read_csv('newsapi_data/scraped_1_18_25.csv')

# run updated search
newsapi_extraction_updated = eliminate_duplicates('student loan forgiveness', api_key, extracted_data)

# get unique sources
unique_sources = get_unique_sources(newsapi_extraction_updated)

# search for unreviewed articles
unreviewed_sources = check_sources(newsapi_extraction_updated)

# update extracted_data
extracted_data = pd.concat([extracted_data, newsapi_extraction_updated], ignore_index=True)

# scrape and concatenate
articles_df = scrape_and_concatenate(newsapi_extraction_updated)

# update scraped data
scraped_data = pd.concat([scraped_data, articles_df], ignore_index=True)

# save dataframes
extracted_data.to_csv('newsapi_data/extraction_1_20_25.csv', index=False)
scraped_data.to_csv('newsapi_data/scraped_1_20_25.csv', index=False)

'''
Additional Data Retrievals - Round 2

- query: 'student loans' (generic - denote v2 in saved dataframes)
'''

# load data
extracted_data = pd.read_csv('newsapi_data/extraction_1_20_25.csv')
scraped_data = pd.read_csv('newsapi_data/scraped_1_20_25.csv')

# run updated search
newsapi_extraction_updated = eliminate_duplicates('student loans', api_key, extracted_data)

# get unique sources
unique_sources = get_unique_sources(newsapi_extraction_updated)

# search for unreviewed articles
unreviewed_sources = check_sources(newsapi_extraction_updated)

# update extracted_data
# extracted_data = pd.concat([extracted_data, newsapi_extraction_updated], ignore_index=True)

# scrape and concatenate
articles_df = scrape_and_concatenate(newsapi_extraction_updated)

# update scraped data
scraped_data = pd.concat([scraped_data, articles_df], ignore_index=True)

# save dataframes
# save updated search to prevent api waiting time
newsapi_extraction_updated.to_csv('newsapi_data/raw_extraction_v2_1_24_25.csv', index=False)
articles_df.to_csv('newsapi_data/scraped_v2_1_24_25.csv', index=False)
    
'''
Additional Data Retrievals - Round 3

- query: 'student loan forgiveness'
'''
# load data - cleaned
data_path = '../cleaning/newsapi_data_cleaned/extraction_cleaned.csv'
extracted_data = pd.read_csv(data_path)

# load data - extracted uncleaned
data_path = 'newsapi_data/extraction_1_20_25.csv'
extracted_data_unclean = pd.read_csv(data_path)

# load data - scraped uncleaned
data_path = 'newsapi_data/scraped_1_20_25.csv'
scraped_data_unclean = pd.read_csv(data_path)

# run updated search
arguments = {'from': '2025-01-25'}
newsapi_extraction_updated = eliminate_duplicates('student loan forgiveness', api_key, extracted_data, arguments, iterate=True)

# get unique sources
unique_sources = get_unique_sources(newsapi_extraction_updated)

# search for unreviewed articles
unreviewed_sources = check_sources(newsapi_extraction_updated)

# update extracted_data
extracted_data = pd.concat([extracted_data_unclean, newsapi_extraction_updated], ignore_index=True)

# scrape and concatenate
articles_df = scrape_and_concatenate(newsapi_extraction_updated)

# update scraped data
scraped_data = pd.concat([scraped_data_unclean, articles_df], ignore_index=True)

# save dataframes
# save updated search to prevent api waiting time
newsapi_extraction_updated.to_csv('newsapi_data/raw_extraction_2_3_25.csv', index=False)
extracted_data.to_csv('newsapi_data/extraction_2_3_25.csv', index=False)
scraped_data.to_csv('newsapi_data/scraped_2_3_25.csv', index=False)

'''
Additional Data Retrievals - Round 4

- query: 'student loans' (generic - denote v2 in saved dataframes)
'''
# load data - cleaned
data_path = '../cleaning/newsapi_data_cleaned/extraction_cleaned.csv'
extracted_data = pd.read_csv(data_path)

# load data - extracted uncleaned
data_path = 'newsapi_data/raw_extraction_v2_1_24_25.csv'
extracted_data_unclean = pd.read_csv(data_path)

# load data - scraped uncleaned
data_path = 'newsapi_data/scraped_v2_1_24_25.csv'
scraped_data_unclean = pd.read_csv(data_path)

# run updated search
arguments = {'from': '2025-01-25'}
newsapi_extraction_updated = eliminate_duplicates('student loans', api_key, extracted_data, arguments, iterate=True)

# get unique sources
unique_sources = get_unique_sources(newsapi_extraction_updated)

# search for unreviewed articles
unreviewed_sources = check_sources(newsapi_extraction_updated)
unreviewed_df = newsapi_extraction_updated[newsapi_extraction_updated['source'].isin(unreviewed_sources)]

# update ineligible
ineligible = create_ineligible()
newsapi_extraction_updated_new = newsapi_extraction_updated[~newsapi_extraction_updated['source'].isin(ineligible)]

# update extracted_data
extracted_data = pd.concat([extracted_data_unclean, newsapi_extraction_updated_new], ignore_index=True)

# scrape and concatenate
articles_df = scrape_and_concatenate(newsapi_extraction_updated_new)

# update scraped data
scraped_data = pd.concat([scraped_data_unclean, articles_df], ignore_index=True)

# save dataframes
# save updated search to prevent api waiting time
newsapi_extraction_updated_new.to_csv('newsapi_data/raw_extraction_v2_2_3_25.csv', index=False)
extracted_data.to_csv('newsapi_data/extraction_v2_2_3_25.csv', index=False)
scraped_data.to_csv('newsapi_data/scraped_v2_2_3_25.csv', index=False)

'''
Additional Data Retrievals - Round 5

- query: 'student loan forgiveness'
- Notes:
    - since going from such a future date, won't need to use the remove_duplicates version
'''
# further arguments
arguments = {'from': '2025-02-04'}

# extract articles with topic 'Student Loan Forgiveness'
newsapi_extraction = extract_newsapi(topic='student loan forgiveness', api_key=api_key, sort_by='popularity', arguments=arguments, iterate=True)

# reformat extracted article information
news_df = format_newsapi(newsapi_extraction)

# get unique sources
unique_sources = get_unique_sources(news_df)

# search for unreviewed articles
unreviewed_sources = check_sources(news_df)

# scrape and concatenate
articles_df = scrape_and_concatenate(news_df)

# save dataframes
news_df.to_csv('newsapi_data/raw_extraction_2_14_25.csv', index=False)
articles_df.to_csv('newsapi_data/scraped_2_14_25.csv', index=False)

'''
Additional Data Retrievals - Round 6

- query: 'student loans'
- Notes:
    - since going from such a future date, won't need to use the remove_duplicates version
'''
# further arguments
arguments = {'from': '2025-02-04'}

# extract articles with topic 'Student Loan Forgiveness'
newsapi_extraction = extract_newsapi(topic='student loans', api_key=api_key, sort_by='popularity', arguments=arguments, iterate=True)

# reformat extracted article information
news_df = format_newsapi(newsapi_extraction)

# get unique sources
unique_sources = get_unique_sources(news_df)

# search for unreviewed articles
unreviewed_sources = check_sources(news_df)

# scrape and concatenate
articles_df = scrape_and_concatenate(news_df)

# save dataframes
news_df.to_csv('newsapi_data/raw_extraction_v2_2_14_25.csv', index=False)
articles_df.to_csv('newsapi_data/scraped_v2_2_14_25.csv', index=False)


'''
Additional Data Retrievals - Training Round

- get data to train bias models
'''

# import cleaned and labeled newsapi data
newsapi_labeled = pd.read_csv('../cleaning/newsapi_data_cleaned/newsapi_labeled.csv')

# look at sources which contain bias information
newsapi_labeled_bias = newsapi_labeled[newsapi_labeled['Bias Specific'].notnull()]
newsapi_labeled_bias.reset_index(drop=True, inplace=True)

# look at biases
labels = newsapi_labeled_bias['Bias Specific'].value_counts()

'''
Center        122
Lean Left      72
Right          33
Lean Right     20
Left           19
'''

# get the top 3 sources for each label type
top_sources_center = newsapi_labeled_bias[newsapi_labeled_bias['Bias Specific']=='Center']['source'].value_counts().nlargest(3).index.tolist()
top_sources_lean_left = newsapi_labeled_bias[newsapi_labeled_bias['Bias Specific']=='Lean Left']['source'].value_counts().nlargest(3).index.tolist()
top_sources_left = newsapi_labeled_bias[newsapi_labeled_bias['Bias Specific']=='Left']['source'].value_counts().nlargest(3).index.tolist()
top_sources_lean_right = newsapi_labeled_bias[newsapi_labeled_bias['Bias Specific']=='Lean Right']['source'].value_counts().nlargest(3).index.tolist()
top_sources_right = newsapi_labeled_bias[newsapi_labeled_bias['Bias Specific']=='Right']['source'].value_counts().nlargest(3).index.tolist()

# get urls for the top sources
top_urls_center = newsapi_labeled_bias[newsapi_labeled_bias['source'].isin(top_sources_center)].drop_duplicates(subset='source')['url'].tolist()
top_urls_lean_left = newsapi_labeled_bias[newsapi_labeled_bias['source'].isin(top_sources_lean_left)].drop_duplicates(subset='source')['url'].tolist()
top_urls_left = newsapi_labeled_bias[newsapi_labeled_bias['source'].isin(top_sources_left)].drop_duplicates(subset='source')['url'].tolist()
top_urls_lean_right = newsapi_labeled_bias[newsapi_labeled_bias['source'].isin(top_sources_lean_right)].drop_duplicates(subset='source')['url'].tolist()
top_urls_right = newsapi_labeled_bias[newsapi_labeled_bias['source'].isin(top_sources_right)].drop_duplicates(subset='source')['url'].tolist()

# extract the top 3 domains for eaach label type
top_domains_center = [extract_domain(url) for url in top_urls_center]
top_domains_lean_left = [extract_domain(url) for url in top_urls_lean_left]
top_domains_left = [extract_domain(url) for url in top_urls_left]
top_domains_lean_right = [extract_domain(url) for url in top_urls_lean_right]
top_domains_right = [extract_domain(url) for url in top_urls_right]

# apply generic function
generic_arguments = {'sortBy': 'publishedAt'}
training_data_center = extract_newsapi_generic(api_key, top_domains_center, generic_arguments, False)
training_data_lean_left = extract_newsapi_generic(api_key, top_domains_lean_left, generic_arguments, False)
training_data_left = extract_newsapi_generic(api_key, top_domains_left, generic_arguments, False)
training_data_lean_right = extract_newsapi_generic(api_key, top_domains_lean_right, generic_arguments, False)
training_data_right = extract_newsapi_generic(api_key, top_domains_right, generic_arguments, False)

# reformat extracted article information
df_center = format_newsapi(training_data_center)
df_lean_left = format_newsapi(training_data_lean_left)
df_left = format_newsapi(training_data_left)
df_lean_right = format_newsapi(training_data_lean_right)
df_right = format_newsapi(training_data_right)

# concatenate the dataframes together
df_training = pd.concat([df_center, df_lean_left], ignore_index=True)
df_training = pd.concat([df_training, df_left], ignore_index=True)
df_training = pd.concat([df_training, df_lean_right], ignore_index=True)
df_training = pd.concat([df_training, df_right], ignore_index=True)

# save dataframe
df_training.to_csv('newsapi_data/training_extraction_2_11_25.csv', index=False)

# check for duplicates with the topic specific data
duplicates = [url for url in df_training['url'].tolist() if url in newsapi_labeled['url'].tolist()]

# scrape the articles
training_scraped = scrape_and_concatenate(df_training)

# save dataframe
training_scraped.to_csv('newsapi_data/training_scraped_2_11_25.csv', index=False)
