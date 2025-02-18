'''
Media Bias Scraping Functions
'''

# import libraries
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import os
import re

def scrape_main_biases():
    '''
    Scrape the main page of AllSides biases to get URLs.
    
    Returns
    -------
    df_body : pandas dataframe
        Contains information about the main page of biases, particularly the URLs.

    '''
    # use selenium and beautifulsoup
    driver = webdriver.Chrome()
    url = 'https://www.allsides.com/media-bias/ratings'
    driver.get(url)
    driver.implicitly_wait(10)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    body = soup.find('tbody')
    base_url = 'https://www.allsides.com'
    body_source_urls = body.find_all(class_='views-field views-field-title source-title')
    body_bias = body.find_all(class_='views-field views-field-field-bias-image')
    body_feedback = body.find_all(class_='getratingval')
    
    body_extractions = {'source': [],
                        'url': [],
                        'bias': [],
                        'community_yes': [],
                        'community_no': [],
                        'community_indicator': []}
    
    # get source and url
    for element in body_source_urls:
        source = element.find('a').text
        url = f"{base_url}{element.find('a')['href']}"
        body_extractions['source'].append(source)
        body_extractions['url'].append(url)
        
    # get bias
    for element in body_bias:
        bias = element.find('a')['href']
        body_extractions['bias'].append(bias.split('/')[-1])
        
    # get community feedback
    for element in body_feedback:
        agree = element.find(class_='agree').text
        disagree = element.find(class_='disagree').text
        feedback = element.find(class_='commtext').text
        body_extractions['community_yes'].append(agree)
        body_extractions['community_no'].append(disagree)
        body_extractions['community_indicator'].append(feedback)
    
    # base data
    df_body = pd.DataFrame(body_extractions)
    
    return df_body

# function to get specific information
def get_specific_bias(source, url, featured_specific, implicit_wait=30):
    '''
    Get the specific information about an AllSide news source.
    
    Parameters
    ----------
    source : string
        Name of the source in AllSides.
    url : string
        URL of the source in AllSides.
    featured_specific : dictionary
        Dictionary to hold the results of the specifics.
    implicit_wait : integer, optional
        Wait time between scrape calls. The default is 30.

    Returns
    -------
    featured_specific : dictionary
        Results of multiple source specifics.
    '''
    
    try:
        # selenium and beautifulsoup scraping
        driver = webdriver.Chrome()
        driver.get(url)
        driver.implicitly_wait(implicit_wait)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        
        # specific political and numerical leanings
        try:
            rating_numeric = soup.find('meter')['value']
        except:
            rating_numeric = '0'
            
        rating_specific = soup.find(class_='bias-rating').text
        featured_specific['Bias Numeric'].append(rating_numeric)
        featured_specific['Bias Specific'].append(rating_specific)
        
        # get table information
        source_information = soup.find(class_='info-table').find_all('tr')
        
        # iterate through table
        for element in source_information:
            element_specifics = element.find_all('td')
            
            # check if specifics contain informatoin
            if len(element_specifics) > 0:
                if element_specifics[0].text == 'Type':
                    featured_specific['Type'].append(element_specifics[1].text)
                elif element_specifics[0].text == 'Region':
                    featured_specific['Region'].append(element_specifics[1].text)
                elif element_specifics[0].text == 'Website':
                    featured_specific['Website'].append(element_specifics[1].find('a')['href'])
                    
        # fill source and url into data structure (only after all other scrapers have succeeded)
        featured_specific['source'].append(source)
        featured_specific['url'].append(url)
                    
        # check and fill with None if data storage has all elements filled
        max_length = max(len(value) for value in featured_specific.values())
        for key, value in featured_specific.items():
            featured_specific[key] = value + [None] * (max_length - len(value))
            
        # close web browser
        driver.close()
    # will return the data gathered upon exception (errored out)
    except:
        print(f'Error with {source}\n')
        driver.close()
        for key in featured_specific:
            # source and url are always provided
            if key == 'source':
                featured_specific[key].append(source)
            elif key == 'url':
                featured_specific[key].append(url)
            else:
                featured_specific[key].append(None)
            
        return featured_specific
    
    return featured_specific

# function to iterate through the sources
def get_bias_detail(df_sources, implicit_wait=30):
    '''
    Iterates through dataframe with sources and urls of AllSides' sources.

    Parameters
    ----------
    df_sources : pandas dataframe
        Dataframe with at least source and url for AllSide's data.
    implicit_wait : integer, optional
        Wait time between scrape calls. The default is 30.

    Returns
    -------
    featured_specific : dictionary
        Scraped specifics of Allsides' sources:
            - source
            - url
            - bias (numeric)
            - bias (specific)
            - type
            - region
            - website
    '''
    
    # create data structure for storage
    featured_specific = {'source': [],
                         'url': [],
                         'Bias Numeric': [],
                         'Bias Specific': [],
                         'Type': [],
                         'Region': [],
                         'Website': []}
    
    for index, row in df_sources.iterrows():
        # get source and url from dataframe
        source = row['source']
        url = row['url']
        
        # print source
        print(f'{source}\n')
        
        # fill data structure
        featured_specific = get_specific_bias(source, url, featured_specific, implicit_wait)
        
    return featured_specific
