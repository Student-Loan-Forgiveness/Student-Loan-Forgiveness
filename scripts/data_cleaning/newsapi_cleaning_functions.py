'''
NEWSAPI Data Cleaning Functions
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

'''
Cleaning Procedures

- Extracted Data (newsapi direct):
    - author:
        - replace() to remove '\n'
        - "unique" ammendments:
            - ', Senior Contributor' to ''
            - ', USA TODAY' to ''
            - ', The Week US' to ''
            - 'editorial staff' to ''
            - 'AP education writer' to ''
            - 'Associated Press' to ''
        - create function remove_email_addresses() to remove any email addresses
        - create function replace_multi_spaces() to replace multi spaces with single spaces
        - replace() to remove '(' and ')'
        - create function remove_links() to remove any website links
        - strip() to remove leading and trailing spaces
        - title() to make name proper
        - if author is blank -> use source
    - content: might ignore in lieu of the article itself
    - description:
        - create function remove_before_delimiter() to remove location and date
        - replace_multi_spaces() to replace multi spaces with single spaces
        - strip() to remove leading and trailing spaces
    - date: apply pandas datetime
    - source: leave
    - title: leave
    
- Scraped Data (full articles scraped):
    - source: leave
    - url: leave
    - paragraph_num: leave
    - paragraph (first pass):
        - create function remove_blank_paragraphs() to remove paragraphs which do not contain words:
            - NoneType
            - string of length 0
            - no word characters
        - replace '\xa0' with ' '
        - apply replace_multi_spaces()
        - apply strip()
        - rerun remove_blank_paragraphs
    - paragraph (second pass):
        - combine all article paragraphs after cleaning
'''

# function remove website links
def remove_links(string):
    result = re.sub(r'http\S+|www\.\S+', '', string)
    return result

# function to remove email addresses
def remove_email_addresses(string):
    result = re.sub(r'\S+@\S+', '', string)
    return result.strip()

# function to remove before delimiter
def remove_before_delimiter(string, delimiter='-- '):
    index = string.find(delimiter)
    if index != -1:
        result = string[index + len(delimiter):]
    else:
        result = string
    
    return result

# function to ammend unique situations
def ammend_unique_author(string):
    # list of unique scenarios (i.e. no general pattern)
    unique_scenarios = [', Senior Contributor',
                        ', USA TODAY',
                        ', The Week US',
                        'editorial staff',
                        'AP education writer',
                        'Associated Press',
                        ' Forbes Staff']
    
    # check if unique scenario present in string
    for scenario in unique_scenarios:
        if scenario in string:
            string = string.replace(scenario, '')
    
    return string

# function to remove duplicate authors
def remove_duplicate_authors(string):
    authors = string.split(', ')
    unique_authors = []
    for author in authors:
        if author not in unique_authors:
            unique_authors.append(author)
            
    return ', '.join(unique_authors)

# function to remove multiple spaces
def remove_multiple_spaces(string):
    return re.sub(r'\s+', ' ', string).strip()
    
# function to clean extracted data
def clean_extracted_data(extracted_data):
    '''
    - author:
        - replace() to remove '\n'
        - "unique" ammendments:
            - ', Senior Contributor' to ''
            - ', USA TODAY' to ''
            - ', The Week US' to ''
            - 'editorial staff' to ''
            - 'AP education writer' to ''
            - 'Associated Press' to ''
            - 'Forbes Staff'
        - create function remove_email_addresses() to remove any email addresses
        - replace() to remove '(' and ')'
        - create function remove_links() to remove any website links
        - strip() to remove leading and trailing spaces
        - title() to make name proper
        - create function replace_multi_spaces() to replace multi spaces with single spaces
        - if author is blank or has zero-length value, use source
        - create function remove_duplicate_authors() to remove repeat authors in single article
        - strip(',') to remove any trailing or leading commas
    - content: leave in lieu of the article itself
    - description:
        - create function remove_before_delimiter() to remove location and date
        - replace_multi_spaces() to replace multi spaces with single spaces
        - strip() to remove leading and trailing spaces
    - date: leave but apply pandas datetime for proper use
    - source: leave
    - title: leave
    '''
    # create copy
    extracted_copy = extracted_data.copy()
    
    # author - remove line breaks
    extracted_copy['author'] = extracted_copy['author'].apply(lambda row: row.replace('\n', '') if isinstance(row, str) else row)
    
    # author - unique ammendments
    extracted_copy['author'] = extracted_copy['author'].apply(lambda row: ammend_unique_author(row) if isinstance(row, str) else row)
    
    # author - remove email addresses
    extracted_copy['author'] = extracted_copy['author'].apply(lambda row: remove_email_addresses(row) if isinstance(row, str) else row)

    # author - remove parenthesis
    extracted_copy['author'] = extracted_copy['author'].apply(lambda row: row.replace('(', '') if isinstance(row, str) else row)
    extracted_copy['author'] = extracted_copy['author'].apply(lambda row: row.replace(')', '') if isinstance(row, str) else row)

    # author - remove links
    extracted_copy['author'] = extracted_copy['author'].apply(lambda row: remove_links(row) if isinstance(row, str) else row)
    
    # author - remove leading and trailing spaces
    extracted_copy['author'] = extracted_copy['author'].apply(lambda row: row.strip() if isinstance(row, str) else row)
    
    # author - title formatting
    extracted_copy['author'] = extracted_copy['author'].apply(lambda row: row.title() if isinstance(row, str) else row)
    
    # author - replace inner multi spaces
    extracted_copy['author'] = extracted_copy['author'].apply(lambda row: remove_multiple_spaces(row) if isinstance(row, str) else row)
    
    # author - replace None or '' with source
    extracted_copy['author'] = extracted_copy['author'].fillna(extracted_data['source'])
    for index, row in extracted_copy.iterrows():
        author = row['author']
        source = row['source']
        if len(author) == 0:
            extracted_copy.loc[index, 'author'] = source
            
    # author - remove duplicate authors
    extracted_copy['author'] = extracted_copy['author'].apply(lambda row: remove_duplicate_authors(row))
    
    # author - remove leading and trailing commas
    extracted_copy['author'] = extracted_copy['author'].apply(lambda row: row.strip(','))
    
    # description - remove location and date
    extracted_copy['description'] = extracted_copy['description'].apply(lambda row: remove_before_delimiter(row) if isinstance(row, str) else row)
    
    # description - replace inner multi spaces
    extracted_copy['description'] = extracted_copy['description'].apply(lambda row: remove_multiple_spaces(row) if isinstance(row, str) else row)
    
    # description - remove leading and trailing spaces
    extracted_copy['description'] = extracted_copy['description'].apply(lambda row: row.strip() if isinstance(row, str) else row)
    
    return extracted_copy

'''
Scraped Data
'''
# function to remove blank paragraphs
def contains_words(string):
    return bool(re.search(r'\w', string))

# function to clean paragraphs in scraped data
def clean_scraped_paragraphs(scraped_data):
    '''
    - source: leave
    - url: leave
    - paragraph (first pass):
        - replace '\xa0' with ' '
        - apply strip()
        - apply replace_multi_spaces()
        - create function contains_words() to help remove paragraphs which do not contain words:
            - string of length 0
            - no word characters
        - recount paragraphs using unique urls
    '''
    
    # initialize cleaned dictionary
    scraped_cleaned = {'source': [],
                       'url': [],
                       'paragraph': [],
                       'paragraph_num': []}
    
    # blank paragraphs - first replace NoneType paragrapsh with zero length strings
    scraped_data['paragraph'] = scraped_data['paragraph'].fillna('')
    
    # replace line space html (\xa0)
    scraped_data['paragraph'] = scraped_data['paragraph'].str.replace('\xa0', ' ')
    
    # apply strip
    scraped_data['paragraph'] = scraped_data['paragraph'].str.strip()
    
    # replace multi spaces with single spaces
    scraped_data['paragraph'] = scraped_data['paragraph'].apply(lambda row: remove_multiple_spaces(row))
    
    # remove blank paragraphs
    for index, row in scraped_data.iterrows():
        source = row['source']
        url = row['url']
        paragraph = row['paragraph']
        
        # paragraph counting
        if (len(scraped_cleaned['paragraph']) == 0) or (url != scraped_cleaned['url'][-1]):
            paragraph_num = 0
        
        if contains_words(paragraph):
            scraped_cleaned['source'].append(source)
            scraped_cleaned['url'].append(url)
            scraped_cleaned['paragraph'].append(paragraph)
            scraped_cleaned['paragraph_num'].append(paragraph_num)
            paragraph_num += 1
            
    # return dataframe of cleaned scraped data
    return pd.DataFrame(scraped_cleaned)

# function to combine paragraphs from unique article urls
def combine_paragraphs(cleaned_scraped):
    # initialize data structure for storage of combined paragraphs
    articles_scraped = {'source': [],
                        'url': [],
                        'article': []}
    
    # unique urls
    unique_urls = cleaned_scraped['url'].unique().tolist()
    
    # iterate through the unique urls
    for url in unique_urls:
        # subset and reset index
        df_subset = cleaned_scraped[cleaned_scraped['url']==url]
        df_subset.reset_index(drop=True, inplace=True)
        
        # get source
        source = df_subset.loc[0, 'source']
        
        # concatenate paragraphs
        article = '\n'.join(df_subset['paragraph'].tolist())
        
        # populate dictionary
        articles_scraped['source'].append(source)
        articles_scraped['url'].append(url)
        articles_scraped['article'].append(article)
        
    # return dataframe of combined articles
    return pd.DataFrame(articles_scraped)
