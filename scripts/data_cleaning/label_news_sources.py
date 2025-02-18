'''
Labeling News Sources
'''

# import libraries
import numpy as np
import pandas as pd
import os

# import newsapi data
newsapi_extraction = pd.read_csv('newsapi_data_cleaned/extraction_cleaned_2_14_25.csv')
newsapi_scraped = pd.read_csv('newsapi_data_cleaned/scraped_cleaned_2_14_25.csv')

# import media bias data
media_bias_data = pd.read_csv('../data_acquisition/allsides_data/final_specific.csv')
media_bias_map = pd.read_csv('../data_acquisition/allsides_data/source_to_source_v2.csv')

'''
Part 1: Media Bias labels

Retain:
    - source (key)
    - Bias Numeric
    - Bias Specific
    
Ignore:
    - Type (all same type of media)
    - Region (mostly the same)
    - Website (no important information for predictive purposes)
    
Strategy:
    1. subset media_bias_data on retain columns
    2. rename media_bias_data "source" column to "source_bias"
    3. merge media_bias_data (subset of retain columns) and media_bias_map
    4. merge newsapi_scraped and media_bias_data - media_bias_map merge
'''

# subset
media_bias_data_subset = media_bias_data[['source', 'Bias Numeric', 'Bias Specific']].copy()

# rename
media_bias_data_subset.rename(columns={'source': 'source_bias'}, inplace=True)

# merge media_bias_data and media_bias_map
media_bias_merge = pd.merge(media_bias_data_subset, media_bias_map, on='source_bias')

# merge newsapi_scraped into above merge
newsapi_scraped_labeled = pd.merge(newsapi_scraped, media_bias_merge, on='source', how='left')

# ensure no duplicates
newsapi_scraped_labeled.drop_duplicates(inplace=True, ignore_index=True)

'''
Part 2: NewsAPI Extraction Labels

Retain:
    - author
    - description (note: would be interesting to see how well description compares to full article)
    - date
    - title
    - search
    - url (key)

Ignore:
    - source (retained under scraped version)
    - content (replaced by actual full article)
    
Strategy:
    1. subset newsapi extraction on retain columns
    2. merge subsetted newsapi extraction with media bias labeled newsapi scraped data on "url"
'''

# subset
newsapi_extraction_subset = newsapi_extraction[['author', 'description', 'date', 'title', 'search', 'url']].copy()

# merge labeled newsapi scraped data with extraction subset
newsapi_labeled = pd.merge(newsapi_scraped_labeled, newsapi_extraction_subset, on='url')

# save final merged dataframe
newsapi_labeled.to_csv('newsapi_data_cleaned/newsapi_labeled_2_14_25.csv', index=False)

'''
Part 3: Labeling Generic Data

- generic data was extracted to help train bias models
- search parameter will be ignored since this is for bias training
'''

# import scraped generic training data
training_scraped = pd.read_csv('newsapi_data_cleaned/training_scraped_cleaned.csv')

# import extraction generic training data
training_extraction = pd.read_csv('newsapi_data_cleaned/training_extraction_cleaned.csv')

# import specific labeled data
newsapi_labeled = pd.read_csv('newsapi_data_cleaned/newsapi_labeled_2_14_25.csv')

# subset extraction data
training_extraction_subset = training_extraction[['author', 'description', 'date', 'title', 'url']].copy()

# merge the newsapi dataframes together
training_merged = pd.merge(training_scraped, training_extraction_subset, on='url')

# subset media bias data
media_bias_subset = newsapi_labeled[['source', 'Bias Numeric', 'Bias Specific']].copy()
media_bias_subset.drop_duplicates(inplace=True)
media_bias_subset.dropna(inplace=True)
media_bias_subset.reset_index(drop=True, inplace=True)

# merge in bias
training_labeled = pd.merge(training_merged, media_bias_subset, on='source')

# save dataframe
training_labeled.to_csv('newsapi_data_cleaned/training_labeled_2_14_25.csv', index=False)
