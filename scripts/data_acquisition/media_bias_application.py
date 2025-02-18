'''
Media Bias Scraping Applications
'''

# import libraries
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import os
import re
import time

# import specific functions
from media_bias_functions import *

'''
Part 1: Main - Featured - Sources from AllSides
'''

# get the main body urls
df_main = scrape_main_biases()
df_main.to_csv('allsides_data/featured_body.csv', index=False)

# get the specific bias data from the main body urls - account for website blocking requests after a number of iterations
featured_main_v1 = get_bias_detail(df_main)
df_featured_main_v1 = pd.DataFrame(featured_main_v1)
df_featured_main_v1.to_csv('allsides_data/featured_data_v1.csv', index=False)

# iterations blocked - account for this
df_main_blocked = df_featured_main_v1[df_featured_main_v1.isnull().any(axis=1)]
featured_main_v2 = get_bias_detail(df_main_blocked)
df_featured_main_v2 = pd.DataFrame(featured_main_v2)
df_featured_main_v2.to_csv('allsides_data/featured_data_v2.csv', index=False)

# iterations blocked - account for this
df_main_blocked = df_featured_main_v2[df_featured_main_v2.isnull().any(axis=1)]
featured_main_v3 = get_bias_detail(df_main_blocked)
df_featured_main_v3 = pd.DataFrame(featured_main_v3)
df_featured_main_v3.to_csv('allsides_data/featured_data_v3.csv', index=False)

# drop the nulls for each version of the iteration
df_v1 = df_featured_main_v1[df_featured_main_v1.notnull().any(axis=1)]
df_v2 = df_featured_main_v2[df_featured_main_v2.notnull().any(axis=1)]
df_v3 = df_featured_main_v3[df_featured_main_v3.notnull().any(axis=1)]

# concatenate the non-null data
df_featured_main = pd.concat([df_v1, df_v2], ignore_index=True)
df_featured_main = pd.concat([df_featured_main, df_v3], ignore_index=True)
df_featured_main.to_csv('allsides_data/final_featured.csv', index=False)

# load to reset when needed
df_featured_main = pd.read_csv('allsides_data/final_featured.csv')

'''
Part 2: Initial Matching of NewsAPI Sources with Featured AllSides' Data

Notes:
    - use scraped sources
    - account for source to source mapping via dictionary into dataframe:
        - "source": list of NewsAPI source names
        - "source_bias": list of respective AllSides source names
'''

# import scraped data
scraped = pd.read_csv('../cleaning/newsapi_data_cleaned/scraped_cleaned.csv')

# manually find source to source mapping
newsapi_sources = scraped['source'].unique().tolist()
bias_sources = df_featured_main['source'].unique().tolist()

# newsapi sources
sources = ['NPR',
           'Forbes',
           'Reason',
           'CBS News',
           'Breitbart News',
           'The Daily Caller',
           'New York Post',
           'ABC News',
           'HuffPost',
           'Fox News',
           'Slate Magazine',
           'CNN',
           'fox6now.com']

# bias sources
sources_bias = ['NPR (Online News)',
                'Forbes',
                'Reason',
                'CBS News (Online)',
                'Breitbart News',
                'The Daily Caller',
                'New York Post (News)',
                'ABC News (Online)',
                'HuffPost',
                'Fox News Digital',
                'Slate',
                'CNN Digital',
                'Fox News Digital']

# create source to source mapping dataframe for main AllSides' sources
source_to_source_main = pd.DataFrame({'source': sources, 'source_bias': sources_bias})

# save dataframe
source_to_source_main.to_csv('allsides_data/source_to_source_v1.csv', index=False)

'''
Part 3: Additional Matching of NewsAPI Sources with Featured AllSides' Data

Notes:
    - follow same structure as Part 2 for the final replacement mapping
    - will need to manually find the urls for this portion
    - create new data structure to include manually found url:
        - "source": list of NewsAPI source names
        - "source_bias": list of respective AllSides source names
        - "url": url with specific AllSides source information
        - NOTE: FOR THE FUNCTION TO WORK, WILL NEED TO RENAME "source_bias" to "source" and "source" to "source_news" 
'''

# unmatched newsapi sources
unmatched_sources = [source for source in newsapi_sources if source not in source_to_source_main['source'].tolist()]

# AllSides Bias Sources
sources_bias = ['Al Jazeera',
                'American Thinker',
                'CNBC',
                'CNET',
                "Investor's Business Daily",
                'Mediaite',
                'ProPublica',
                'Richmond Times Dispatch',
                'RollingStone.com',
                'The Denver Post',
                'The Philadelphia Inquirer',
                'The Week',
                'The Nation',
                'Time Magazine',
                'USA TODAY',
                'Vox',
                'Washington Free Beacon',
                'Washington Monthly',
                'WND',
                'CNN Digital',
                'New York Magazine',
                'Raw Story',
                'Common Dreams',
                'WCVB',
                'The Advocate',
                'Yahoo News',
                'New Republic',
                'The Daily Signal',
                'Boston Herald',
                'The Roanoke Times',
                'NewsBreak',
                'The Western Journal',
                'The Root',
                'FFX Now',
                'Lad Bible',
                'Catholic News Agency',
                'Parkersburg News and Sentinel',
                'Check Your Fact',
                'Honolulu Star-Advertiser',
                'National Post',
                'The Daily Dot',
                'PinkNews',
                'CNN Digital',
                'New York Magazine',
                'Raw Story',
                'Common Dreams',
                'WCVB',
                'The Advocate']

# AllSides Bias URLs
urls = ['https://www.allsides.com/news-source/al-jazeera-media-bias',
        'https://www.allsides.com/news-source/american-thinker',
        'https://www.allsides.com/news-source/cnbc',
        'https://www.allsides.com/news-source/cnet',
        'https://www.allsides.com/news-source/investors-business-daily',
        'https://www.allsides.com/news-source/mediaite-bias',
        'https://www.allsides.com/news-source/propublica',
        'https://www.allsides.com/news-source/richmond-times-dispatch',
        'https://www.allsides.com/news-source/rolling-stone',
        'https://www.allsides.com/news-source/denver-post',
        'https://www.allsides.com/news-source/phillycom',
        'https://www.allsides.com/news-source/the-week-bias',
        'https://www.allsides.com/news-source/nation-media-bias',
        'https://www.allsides.com/news-source/time-magazine-news-media-bias',
        'https://www.allsides.com/news-source/usa-today-media-bias',
        'https://www.allsides.com/news-source/vox-news-media-bias',
        'https://www.allsides.com/news-source/washington-free-beacon',
        'https://www.allsides.com/news-source/washington-monthly',
        'https://www.allsides.com/news-source/wnd-media-bias',
        'https://www.allsides.com/news-source/cnn-media-bias',
        'https://www.allsides.com/news-source/new-york-magazine',
        'https://www.allsides.com/news-source/raw-story',
        'https://www.allsides.com/news-source/common-dreams-media-bias',
        'https://www.allsides.com/news-source/wcvb-media-bias',
        'https://www.allsides.com/news-source/advocate-media-bias-1',
        'https://www.allsides.com/news-source/yahoo-news-media-bias',
        'https://www.allsides.com/news-source/new-republic',
        'https://www.allsides.com/news-source/daily-signal',
        'https://www.allsides.com/news-source/boston-herald-media-bias',
        'https://www.allsides.com/news-source/roanoke-times-media-bias',
        'https://www.allsides.com/news-source/newsbreak-media-bias',
        'https://www.allsides.com/news-source/western-journalism',
        'https://www.allsides.com/news-source/root',
        'https://www.allsides.com/news-source/ffx-now-media-bias',
        'https://www.allsides.com/news-source/lad-bible-media-bias',
        'https://www.allsides.com/news-source/catholic-news-agency-media-bias',
        'https://www.allsides.com/news-source/parkersburg-news-and-sentinel-media-bias',
        'https://www.allsides.com/news-source/check-your-fact-media-bias',
        'https://www.allsides.com/news-source/honolulu-star-advertiser-media-bias',
        'https://www.allsides.com/news-source/national-post-media-bias',
        'https://www.allsides.com/news-source/daily-dot-media-bias',
        'https://www.allsides.com/news-source/pinknews-media-bias',
        'https://www.allsides.com/news-source/cnn-media-bias',
        'https://www.allsides.com/news-source/new-york-magazine',
        'https://www.allsides.com/news-source/raw-story',
        'https://www.allsides.com/news-source/common-dreams-media-bias',
        'https://www.allsides.com/news-source/wcvb-media-bias',
        'https://www.allsides.com/news-source/advocate-media-bias-1']

# NewsAPI Sources
sources = ['Al Jazeera English',
           'Americanthinker.com',
           'CNBC',
           'CNET',
           "Investor's Business Daily",
           'Mediaite',
           'ProPublica',
           'Richmond.com',
           'Rolling Stone',
           'The Denver Post',
           'The Philadelphia Inquirer',
           'The Week',
           'Thenation.com',
           'Time',
           'USA Today',
           'Vox',
           'Washington Free Beacon',
           'Washington Monthly',
           'Wnd.com',
           'CNN',
           'New York Magazine',
           'Raw Story',
           'Common Dreams',
           'WCVB Boston',
           'The Advocate',
           'Yahoo Entertainment',
           'The New Republic',
           'Daily Signal',
           'Boston Herald',
           'Roanoke Times',
           'Newsbreak.com',
           'Westernjournal.com',
           'The Root',
           'ARLnow',
           'LADbible',
           'Catholicnewsagency.com',
           'Parkersburg News',
           'Checkyourfact.com',
           'Honolulu Star-Advertiser',
           'National Post',
           'The Daily Dot',
           'Thepinknews.com',
           'CNN',
           'New York Magazine',
           'Raw Story',
           'Common Dreams',
           'WCVB Boston',
           'The Advocate']

# create source to source mapping dataframe for additional AllSides' sources
source_to_source_additional = pd.DataFrame({'source': sources, 'source_bias': sources_bias})
source_to_source = pd.concat([source_to_source_main, source_to_source_additional], ignore_index=True)

# save dataframe
source_to_source.to_csv('allsides_data/source_to_source_v2.csv', index=False)

# save dataframe
source_to_source_main.to_csv('allsides_data/source_to_source_v1.csv', index=False)

# create dataframe to run through the function
df_additional = pd.DataFrame({'source': sources_bias, 'url': urls})

# initial specific scraping run
specifics_additional = get_bias_detail(df_additional)
specifics_additional_df = pd.DataFrame(specifics_additional)

# check for successes and subsequent reruns
success_df = specifics_additional_df[specifics_additional_df['Bias Numeric'].notnull()]
rerun_df = specifics_additional_df[specifics_additional_df['Bias Numeric'].isnull()]
success_df.reset_index(drop=True, inplace=True)
rerun_df.reset_index(drop=True, inplace=True)

# create copy of successful extraction for the while loop
success_rerun_df = success_df.copy()

# while loop to finish specifics - include wait time to allow calls to website to cool down AND max iterations
iterations = 0
while (rerun_df.size != 0) and (iterations <= 5):
    # additional specific scraping reruns
    specifics_rerun = get_bias_detail(rerun_df)
    specifics_rerun_df = pd.DataFrame(specifics_rerun)
    
    # check for successes and subsequent reruns
    success_rerun_iter_df = specifics_rerun_df[specifics_rerun_df['Bias Numeric'].notnull()]
    rerun_df = specifics_rerun_df[specifics_rerun_df['Bias Numeric'].isnull()]
    success_rerun_iter_df.reset_index(drop=True, inplace=True)
    rerun_df.reset_index(drop=True, inplace=True)
    
    # concatenate succeses
    success_rerun_df = pd.concat([success_rerun_df, success_rerun_iter_df], ignore_index=True)
    
    # add iterations
    iterations += 1
    
    # print iteration
    print(f'ITERATION: {iterations}\n')
    
    # wait time for call cool down
    time.sleep(60)
    
# save additional specific information
success_rerun_df.to_csv('allsides_data/final_additional.csv', index=False)

# concatenate the specific information
final_specific = pd.concat([df_featured_main, success_rerun_df], ignore_index=True)

# ensure Bias Numeric is numerical type
final_specific['Bias Numeric'] = final_specific['Bias Numeric'].astype(float)

# ensure no duplicates
final_specific.drop_duplicates(subset='source', inplace=True, keep='last', ignore_index=True)

# save dataframe
final_specific.to_csv('allsides_data/final_specific.csv', index=False)
