'''
NEWSAPI Data Cleaning Application
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# import specific functions
from newsapi_cleaning_functions import *

'''
Part 1: Extraction Cleaning
'''

# import extracted data - "student loan forgiveness"
extraction_v1_1 = pd.read_csv('../data_acquisition/newsapi_data/extraction_1_18_25.csv')
extraction_v1_2 = pd.read_csv('../data_acquisition/newsapi_data/extraction_1_20_25.csv')
extraction_v1_3 = pd.read_csv('../data_acquisition/newsapi_data/raw_extraction_2_3_25.csv')
extraction_v1_4 = pd.read_csv('../data_acquisition/newsapi_data/raw_extraction_2_14_25.csv')

# import extracted data - "student loans"
extraction_v2_1 = pd.read_csv('../data_acquisition/newsapi_data/raw_extraction_v2_1_24_25.csv')
extraction_v2_2 = pd.read_csv('../data_acquisition/newsapi_data/raw_extraction_v2_2_3_25.csv')
extraction_v2_3 = pd.read_csv('../data_acquisition/newsapi_data/raw_extraction_v2_2_14_25.csv')

# concatenate extracted data - "student loan forgiveness"
extraction_v1 = pd.concat([extraction_v1_1, extraction_v1_2], ignore_index=True)
extraction_v1 = pd.concat([extraction_v1, extraction_v1_3], ignore_index=True)
extraction_v1 = pd.concat([extraction_v1, extraction_v1_4], ignore_index=True)

# concatenate extracted data - "student loans"
extraction_v2 = pd.concat([extraction_v2_1, extraction_v2_2], ignore_index=True)
extraction_v2 = pd.concat([extraction_v2, extraction_v2_3], ignore_index=True)

# add column for search parameter label
extraction_v1['search'] = 'student_loan_forgiveness'
extraction_v2['search'] = 'student_loans'

# concatenate the results from both searches
extraction = pd.concat([extraction_v1, extraction_v2], ignore_index=True)

# ensure no duplicate urls (i.e. articles)
extraction.drop_duplicates(inplace=True, subset='url')
extraction.reset_index(drop=True, inplace=True)

# save extraction dataframe
extraction.to_csv('newsapi_data_cleaned/extraction_raw_2_14_25.csv', index=False)

# apply extraction cleaning function
cleaned_extraction = clean_extracted_data(extraction)

# save cleaned extraction dataframe
cleaned_extraction.to_csv('newsapi_data_cleaned/extraction_cleaned_2_14_25.csv', index=False)

'''
Part 2: Scraped Extraction Cleaning

Notes:
    - clean scraped data first as this combines the paragraphs of each URL
    - concatenate final results, can skip the search label as this will be linked via URL to extracted
'''

# import scraped data
scraped_v1_1 = pd.read_csv('../data_acquisition/newsapi_data/scraped_1_18_25.csv')
scraped_v1_2 = pd.read_csv('../data_acquisition/newsapi_data/scraped_1_20_25.csv')
scraped_v1_3 = pd.read_csv('../data_acquisition/newsapi_data/scraped_2_3_25.csv')
scraped_v1_4 = pd.read_csv('../data_acquisition/newsapi_data/scraped_2_14_25.csv')
scraped_v2_1 = pd.read_csv('../data_acquisition/newsapi_data/scraped_v2_1_24_25.csv')
scraped_v2_2 = pd.read_csv('../data_acquisition/newsapi_data/scraped_v2_2_3_25.csv')
scraped_v2_3 = pd.read_csv('../data_acquisition/newsapi_data/scraped_v2_2_14_25.csv')

# apply function to clean scraped data
cleaned_scraped_1 = clean_scraped_paragraphs(scraped_v1_1)
cleaned_scraped_2 = clean_scraped_paragraphs(scraped_v1_2)
cleaned_scraped_3 = clean_scraped_paragraphs(scraped_v1_3)
cleaned_scraped_4 = clean_scraped_paragraphs(scraped_v1_4)
cleaned_scraped_5 = clean_scraped_paragraphs(scraped_v2_1)
cleaned_scraped_6 = clean_scraped_paragraphs(scraped_v2_2)
cleaned_scraped_7 = clean_scraped_paragraphs(scraped_v2_3)

# apply function to combine paragraphs within cleaned scraped data
articles_scraped_1 = combine_paragraphs(cleaned_scraped_1)
articles_scraped_2 = combine_paragraphs(cleaned_scraped_2)
articles_scraped_3 = combine_paragraphs(cleaned_scraped_3)
articles_scraped_4 = combine_paragraphs(cleaned_scraped_4)
articles_scraped_5 = combine_paragraphs(cleaned_scraped_5)
articles_scraped_6 = combine_paragraphs(cleaned_scraped_6)
articles_scraped_7 = combine_paragraphs(cleaned_scraped_7)

# concatenate the cleaned scraped data with articles combined
scraped = pd.concat([articles_scraped_1, articles_scraped_2], ignore_index=True)
scraped = pd.concat([scraped, articles_scraped_3], ignore_index=True)
scraped = pd.concat([scraped, articles_scraped_4], ignore_index=True)
scraped = pd.concat([scraped, articles_scraped_5], ignore_index=True)
scraped = pd.concat([scraped, articles_scraped_6], ignore_index=True)
scraped = pd.concat([scraped, articles_scraped_7], ignore_index=True)

# ensure no duplicate urls (i.e. articles) - drop last as that will be the most updated articles
scraped.drop_duplicates(inplace=True, subset='url', keep='last')
scraped.reset_index(drop=True, inplace=True)

# save cleaned scraped dataframe
scraped.to_csv('newsapi_data_cleaned/scraped_cleaned_2_14_25.csv', index=False)

'''
Part 3: Cleaning Generic Data

- generic data was extracted to help train bias models
'''

# import data
training_extraction = pd.read_csv('../data_acquisition/newsapi_data/training_extraction_2_11_25.csv')
training_scraped = pd.read_csv('../data_acquisition/newsapi_data/training_scraped_2_11_25.csv')

# clean extraction
training_extraction_cleaned = clean_extracted_data(training_extraction)

# clean scraped
training_scraped_cleaned = clean_scraped_paragraphs(training_scraped)

# combine paragraphs 
training_articles_scraped = combine_paragraphs(training_scraped_cleaned)

# save dataframes
training_extraction_cleaned.to_csv('newsapi_data_cleaned/training_extraction_cleaned.csv', index=False)
training_articles_scraped.to_csv('newsapi_data_cleaned/training_scraped_cleaned.csv', index=False)
