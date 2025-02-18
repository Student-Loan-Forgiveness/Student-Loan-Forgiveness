'''
NEWSAPI and Article Webscrapers
'''

# import libraries
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import datetime
import os

# function to extract articles with more parameters (good for additional attempts)
def extract_newsapi(topic, api_key, sort_by='popularity', arguments=None, iterate=True):
    '''
    Parameters
    ----------
    topic : string
        Keywords or phrases.
    api_key : string
        NewsAPI key.
    sort_by : string, optional
        The order to sort articles in. The default is 'popularity'. Options include:
            - 'relevancy': articles more closely related to keywords.
            - 'popularity': articles from popular sources and publishers come first.
            - 'publishedAt': newest articles comes first.
    arguments : dictionary, optional
        Dictionary of additional API keys and values.
    iterate : boolean, optional
        If results extend past the maximum (100), will iterate through the pages of results. The default is True.

    Returns
    -------
    articles : list
        List of dictionary newsapi article results.
    '''
    # base url
    base_url = 'https://newsapi.org/v2/everything'
    parameters = {'apiKey': api_key,
                  'q': topic,
                  'sortBy': sort_by,
                  'page': 1}
    
    # additional arguments
    if arguments:
        parameters.update(arguments)
        
    # print parameters
    print(f'Parameters: {parameters}\n')
        
    # call api
    response = requests.get(base_url, parameters)
    
    # initiate articles for status code errors
    articles = None
    
    # confirm call worked
    if response.status_code == 200:
        # jsonify the response
        call_data = response.json()
        
        # extract articles
        articles = call_data['articles']
    else:
        print(f'Error... Status Code: {response.status_code}')
        return articles
    
    # get total number of results
    total_results = call_data['totalResults']
    
    # print total results
    print(f'Total Results: {total_results}')
    
    # iteration parameter
    if (iterate) and (total_results > 100):
        # number of pages
        pages = int(total_results / 100) + 2
        for page in range(2, pages):
            # update page in parameters
            parameters['page'] = page
            
            # call api
            response = requests.get(base_url, parameters)
            
            # confirm call worked
            if response.status_code == 200:
                # jsonify the response
                call_data = response.json()
                
                # extract articles
                articles += call_data['articles']
            else:
                print(f'Error... Status Code: {response.status_code}')
                return articles
    
    return articles

# function to gather generic articles (for bias classification)
def extract_newsapi_generic(api_key, domains, arguments=None, iterate=False):
    # endpoint
    base_url = 'https://newsapi.org/v2/everything'
    
    # parameters
    parameters = {'apiKey': api_key,
                  'page': 1}
    
    # update parameters with arguments
    if arguments:
        parameters.update(arguments)
    
    # update parameters with domains
    domains_string = ','.join(domains)
    parameters['domains'] = domains_string
    
    # print parameters
    print(f'Parameters: {parameters}\n')
        
    # call api
    response = requests.get(base_url, parameters)
    
    # initiate articles for status code errors
    articles = None
    
    # confirm call worked
    if response.status_code == 200:
        # jsonify the response
        call_data = response.json()
        
        # extract articles
        articles = call_data['articles']
    else:
        print(f'Error... Status Code: {response.status_code}')
        return articles
    
    # get total number of results
    total_results = call_data['totalResults']
    
    # print total results
    print(f'Total Results: {total_results}')
    
    # iteration parameter
    if (iterate) and (total_results > 100):
        # number of pages
        pages = int(total_results / 100) + 2
        for page in range(2, pages):
            # update page in parameters
            parameters['page'] = page
            
            # call api
            response = requests.get(base_url, parameters)
            
            # confirm call worked
            if response.status_code == 200:
                # jsonify the response
                call_data = response.json()
                
                # extract articles
                articles += call_data['articles']
            else:
                print(f'Error... Status Code: {response.status_code}')
                return articles
    
    return articles

# function to extract domain
def extract_domain(url):
    pattern = re.compile(r'https://([^/]+)/')
    match = pattern.search(url)
    if match:
        extracted = match.group(1)
        extracted_domain = extracted.replace('www.', '')
    else:
        extracted_domain = None
        
    return extracted_domain

# function to create ineligible sources
def create_ineligible():
    '''
    Creates list of ineligible sources for scraping (403, improper structure, or undesired)

    Returns
    -------
    ineligible : list
        Ineligible or undesired sources.
    '''
    
    # ineligible sources (403 - Forbidden, improper structure, or undesired)
    ineligible = ['Business Insider',
                  'Biztoc.com',
                  'Naturalnews.com',
                  'Newsweek',
                  'Thegatewaypundit.com',
                  'Financial Post',
                  'The Hill',
                  'TheBlaze',
                  'NBC News',
                  'Inside Higher Ed',
                  'Wonkette.com',
                  'International Business Times',
                  'VOA News',
                  'Memeorandum.com',
                  'Dianeravitch.net',
                  'Legalinsurrection.com',
                  'Typepad.com',
                  'Whitecoatinvestor.com',
                  'Activistpost.com',
                  'STLtoday.com',
                  'TODAY',
                  'The Times of India',
                  'Katescreativespace.com',
                  'Freerepublic.com',
                  'PBS',
                  'Kevinmd.com',
                  'Stateline.org',
                  'Lewrockwell.com',
                  'Marca',
                  'Scotusblog.com',
                  'seattlepi.com',
                  'Newser',
                  'Slashdot.org',
                  'WDIV ClickOnDetroit',
                  'WPLG Local 10',
                  'WJXT News4JAX',
                  'WKMG News 6 & ClickOrlando',
                  'KPRC Click2Houston',
                  'WMUR Manchester',
                  'Thesocietypages.org',
                  'Lawyersgunsmoneyblog.com',
                  'The Federalist',
                  'Affordanything.com',
                  'Antaranews.com',
                  'Japansubculture.com',
                  'The Irish Times',
                  'New Zealand Herald',
                  'The Japan Times',
                  'Asbury Park Press',
                  'BBC News',
                  'Bangkok Post',
                  'Japan Today',
                  'Buzzfeed',
                  'CBC News',
                  'Decider',
                  'The New York Review of Books',
                  'Thetakeoffnap.com',
                  'soompi',
                  'Economictimes.com',
                  'Bleeding Cool News',
                  'The Online Citizen',
                  'Digital Trends',
                  'PC Gamer',
                  "Investor's Business Daily",
                  'GoDanRiver.com',
                  'Hospitality Net',
                  'ZDNet',
                  'CounterPunch',
                  'Lse.ac.uk',
                  'Globalresearch.ca',
                  'TheStranger.com',
                  'The Conversation Africa',
                  'Resilience',
                  'Wikipedia.org',
                  'Justintadlock.com',
                  'Thechronicle.com.gh',
                  'ETF Daily News',
                  'Exblog.jp',
                  'The Philadelphia Inquirer',
                  'Honolulu Star-Advertiser',
                  'Milliondollarjourney.com',
                  'Fark.com',
                  'Rolling Stone',
                  'Moonbattery.com',
                  'Protothema.gr',
                  'Finextra',
                  'Checkyourfact.com',
                  'National Post',
                  'Monevator.com',
                  'Mcmansionhell.com',
                  'Thepinknews.com',
                  'Project Syndicate',
                  'GamesRadar+',
                  'Hoover.org',
                  'Politicalwire.com',
                  'Commercial Observer',
                  'The Daily Dot',
                  'Publicknowledge.org',
                  'Ynab.com',
                  'Finovate.com',
                  'Johnaugust.com',
                  'CinemaBlend',
                  'Wattsupwiththat.com',
                  'Sightunseen.com',
                  'Thefreedictionary.com',
                  'The Atlantic',
                  'Cheezburger.com',
                  'Thoughtcatalog.com',
                  'National Institutes of Health',
                  'The New Yorker',
                  'MakeUseOf',
                  'Poynter',
                  'Rogerebert.com',
                  'Snopes.com',
                  'Foreign Policy',
                  'Digital Journal',
                  'NDTV News',
                  'Fast Company',
                  'Lesswrong.com',
                  'Dailyutahchronicle.com',
                  'Courrier International',
                  'RT',
                  'menshealth.com',
                  'Whyevolutionistrue.com',
                  'Theradavist.com',
                  'Knowyourmeme.com',
                  'Voxeurop.eu',
                  'Ms. Magazine',
                  'Android Headlines',
                  'Theartblog.org',
                  'Celebitchy.com',
                  'ABC News (AU)',
                  'Steamykitchen.com',
                  'Andscape.com',
                  'WFTV Orlando',
                  'Theeverygirl.com']
    
    return ineligible

# function to turn article extraction into dataframe
def format_newsapi(articles):
    '''
    Parameters
    ----------
    articles : list (of dictionary data types)
        Result from extract_newsapi(). Article dictionaries from a newsapi extraction call.

    Returns
    -------
    pandas dataframe
        Reformats newsapi extracted article information into a dataframe, ignoring removed articles.
    '''
    
    # call ineligible list
    ineligible = create_ineligible()
    
    # initialize dictionary
    article_dict = {'author': [],
                    'content': [],
                    'description': [],
                    'date': [],
                    'source': [],
                    'title': [],
                    'url': []}
    
    # iterate through articles
    for article in articles:
        # gather information
        author = article['author']
        content = article['content']
        description = article['description']
        date = article['publishedAt']
        source = article['source']['name']
        title = article['title']
        url = article['url'].strip()
        
        # check if article is a duplicate or is null (removed)
        if (url in article_dict['url']) or (source == '[Removed]') or (source in ineligible):
            continue
        else:
            article_dict['author'].append(author)
            article_dict['content'].append(content)
            article_dict['description'].append(description)
            article_dict['date'].append(date)
            article_dict['source'].append(source)
            article_dict['title'].append(title)
            article_dict['url'].append(url)
            
    return pd.DataFrame(article_dict)

# function to get unique unique sources (preliminary purposes)
def get_unique_sources(news_df):
    '''
    Parameters
    ----------
    news_df : pandas dataframe
        Result from format_newsapi(). The reformatted article dataframe.

    Returns
    -------
    unique_sources : pandas dataframe
        Unique article website sources along with a url. For use in creating a webscraping template for the article websites.
    '''
    
    # remove duplicate sources
    unique_sources = news_df.drop_duplicates(subset='source', ignore_index=True)
    
    # isolate source and url of article
    unique_sources = unique_sources[['source', 'url']]
    
    return unique_sources

# function to create soup object
def create_soup(url):
    '''
    Parameters
    ----------
    url : string
        URL for the website to create a BeautifulSoup object of.

    Returns
    -------
    soup : BeautifulSoup object
        Given a status_code of 200, the BeautifulSoup object representing the URL.

    '''
    # request page
    page = requests.get(url)
    
    # return None if page cannot be scraped
    if page.status_code != 200:
        return None
    
    # create BeautifulSoup object
    soup = BeautifulSoup(page.text, 'lxml')
    
    return soup

# function to initialize content dictionary
def init_content_dict():
    '''
    Returns
    -------
    content_dict : dictionary
        Initialized dictionary to store the scraped results from news article websites.

    '''
    # initialize content dictionary
    content_dict = {'source': [],
                    'url': [],
                    'paragraph': [],
                    'paragraph_num': []}
    
    return content_dict

# function to populate content dictionary
def populate_content(source, url, paragraphs):
    '''
    Parameters
    ----------
    source : string
        News article website name.
    url : string
        Specific news article from source.
    paragraphs : list
        Each list contains the text from each article body paragraph scraped.

    Returns
    -------
    pandas dataframe
        Dataframe of the scraped articles' body paragraphs.
    '''
    
    # initialize content dictionary
    content_dict = init_content_dict()
    
    # populate content dictionary
    for para_num, para_content in enumerate(paragraphs):
        content_dict['source'].append(source)
        content_dict['url'].append(url)
        content_dict['paragraph'].append(para_content)
        content_dict['paragraph_num'].append(para_num)
        
    return pd.DataFrame(content_dict)

# function to eliminate duplicates for updated searches advanced
def eliminate_duplicates(topic, api_key, extracted_data, arguments, iterate=True):
    '''
    Eliminates duplicate articles from previous extractions and returns dataframe version of extraction.
    
    Parameters
    ----------
    topic : string
        Keywords or phrases.
    api_key : string
        NewsAPI key.
    extracted_data : pandas dataframe
        Previously extracted and formatted NewsAPI data in. Used to prevent duplicate scrapes.
    arguments : dictionary, optional
        Dictionary of additional API keys and values.
    iterate : boolean, optional
        If results extend past the maximum (100), will iterate through the pages of results. The default is True.

    Returns
    -------
    new_extraction : pandas dataframe
        Updated extracted data.

    '''
    # get current urls in data
    current_urls = extracted_data['url'].unique().tolist()
    
    # perform new search
    new_search = extract_newsapi(topic, api_key, sort_by='popularity', arguments=arguments, iterate=iterate)
    
    # reformat extracted article information
    news_df = format_newsapi(new_search)
    
    # new urls from search
    new_urls = news_df['url'].unique().tolist()
    
    # compare to get new urls
    new_extraction_urls = list(set(new_urls) - set(current_urls))
    
    # get unique extractions
    new_extraction = news_df[news_df['url'].isin(new_extraction_urls)]
    
    # reset index
    new_extraction.reset_index(drop=True, inplace=True)
    
    return new_extraction

# scraper for TechDirt
def scrape_techdirt(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='postbody')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)
    
# scraper for Nakedcapitalism.com
def scrape_nakedcapitalism(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None

    # isolate article body and paragraphs
    body = soup.find(class_='pf-content')
    paragraphs = [para.text for para in body.find_all('p')]
    # break at '_____________' to split between article body and comments
    split = paragraphs.index('_____________')
    article_paragraphs = paragraphs[:split]
    
    # populate dictionary
    return populate_content(source, url, article_paragraphs)
    
# scraper for forbes
def scrape_forbes(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-body fs-article fs-responsive-text current-article')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for livescience
def scrape_livescience(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='text-copy bodyCopy auto')
    paragraphs  = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)
        
# scraper for yahoo
def scrape_yahoo(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='body yf-tsvcyu')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for msnbc
def scrape_msnbc(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-body__content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for norml
def scrape_norml(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for reason
def scrape_reason(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for washington monthly
def scrape_washingtonmonthly(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Lifesciencesworld.com
def scrape_lifesciencesworld(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry-content clear')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for globe news wire
def scrape_globenewswire(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='main-body-container article-body')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for investing
def scrape_investing(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article_container')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for POPSUGAR
def scrape_popsugar(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='NodeArticlestyles__ObscuredContentWrapper-sc-1rwhog1-6 eCcsQv')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for nypost
def scrape_nypost(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='single__content entry-content m-bottom')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Vox
def scrape_vox(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='_1agbrixy')
    paragraphs = [para.text for para in body.find_all(class_='duet--article--article-body-component')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for The Denver Post
def scrape_denverpost(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='body-copy')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for fox news
def scrape_foxnews(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-body')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for fox 9
def scrape_fox9(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for newsbreak
def scrape_newsbreak(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='jsx-777308592 jsx-3381000567 content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for CNET
def scrape_cnet(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='c-pageArticle_content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Time
def scrape_time(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='flex w-full max-w-article-body-centered flex-col lg:max-w-article-body')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for NPR
def scrape_npr(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='storytext storylocation linkLocation')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Al Jazeera English
def scrape_aljazeeraenglish(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='wysiwyg wysiwyg--all-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for The New Republic
def scrape_newrepublic(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-body-wrap')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Americanthinker.com
def scrape_americanthinker(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for CBS News
def scrape_cbs(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='content__body')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for USA Today
def scrape_usatoday(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='gnt_ar_b')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Daily Signal
def scrape_dailysignal(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='tds-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Breitbart News
def scrape_breitbart(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for The Week Magazine
def scrape_theweekmag(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article__body')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for The Daily Caller
def scrape_dailycaller(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-content mb-2 pb-2 tracking-tight')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Boston Herald
def scrape_bostonherald(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-body')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for CNBC
def scrape_cnbc(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='ArticleBody-articleBody')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for NerdWallet
def scrape_nerdwallet(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='_2_Pyfm _1gun6R')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Wnd.com
def scrape_wnd(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='dynamic-entry-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Washington Free Beacon
def scrape_washfreebeacon(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for ABC News
def scrape_abc(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='xvlfx ZRifP TKoO eaKKC EcdEg bOdfO qXhdi NFNeu UyHES')
    paragraphs = [para.text for para in body.find_all(class_='EkqkG IGXmU nlgHS yuUao lqtkC TjIXL aGjvy')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for HuffPost
def scrape_huffpost(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry__content-list-container js-cet-unit-buzz_body')
    paragraphs = [para.text for para in body.find_all(class_='primary-cli cli cli-text')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for The Star Online
def scrape_thestar(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='story bot-15 relative')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Abajournal.com
def scrape_abajournal(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='col-xs-12 col-md-8')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Roanoke Times
def scrape_roanoketimes(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='lee-track-in-article asset-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Mediaite
def scrape_mediaite(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='o-post-wrap')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Westernjournal.com
def scrape_westernjournal(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='single-post')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for The Root
def scrape_theroot(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='sc-r43lxo-1 cwnrYD')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for ARLnow
def scrape_arlnow(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry-content pt-6 px-4 content-post')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Business Today
def scrape_businesstoday(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='text-formatted field field--name-body field--type-text-with-summary field--label-hidden field__item')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for BusinessLine
def scrape_businessline(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='contentbody verticle-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Parkersburg News
def scrape_parkersburg(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Richmond.com
def scrape_richmond(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='lee-text-row')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Thenation.com
def scrape_thenation(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='blocks-wrapper type-standard-article is-layout-flow')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for ProPublica
def scrape_propublica(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-body')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for blogTO
def scrape_blogto(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='rich-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for LADbible
def scrape_ladbible(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='content-template_body__RWRhb')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Catholicnewsagency.com
def scrape_catholicnewsagency(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='col post-content content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for mindbodygreen.com
def scrape_mindbodygreen(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='sc-11rlt5o-3 lgLchI')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for The Punch
def scrape_thepunch(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='post-content article-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Patheos
def scrape_patheos(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry-content clearfix')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Elearningindustry.com
def scrape_elearningindustry(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry-content js-trackable')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for The Cut
def scrape_thecut(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-content inline')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for The Portland Mercury
def scrape_portlandmercury(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='component article-body my-1 fs-3 ff-serif fw-light')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Slate Magazine
def scrape_slatemag(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article__content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for pymnts.com
def scrape_pymnts(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='single lh-article mt-1 lnk-article')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Livemint
def scrape_livemint(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='storyPage_storyContent__m_MYl')
    paragraphs = [para.text for para in body.find_all(class_='storyParagraph')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Rolling Out
def scrape_rollingout(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='flex-1 flex flex-col gap-3 px-4 max-w-3xl mx-auto w-full pt-1')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for CNN
def scrape_cnn(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article__content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Twistedsifter.com
def scrape_twistedsifter(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Askamanager.com
def scrape_askamanager(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for New York Magazine
def scrape_newyorkmag(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-content inline')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Raw Story
def scrape_rawstory(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='body-description')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Pajiba.com
def scrape_pajiba(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find('div', id='EntryBody')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Common Dreams
def scrape_commondreams(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='body-description')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Hollywood Life
def scrape_hollywoodlife(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Chicago Reader
def scrape_chicagoreader(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for PMcarpenter.com
def scrape_pmcarpenter(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='post-content__body stSKMK')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for WCVB Boston
def scrape_wcvbboston(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-content--body-text')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for WSOC Charlotte
def scrape_wsoccharlotte(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='default__ArticleBody-sc-tl066j-1 fQmwWk article-body-wrapper')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for WSB Atlanta
def scrape_wsbatlanta(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='default__ArticleBody-sc-tl066j-1 fQmwWk article-body-wrapper')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for fox6now.com
def scrape_fox6now(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='article-body')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for The Advocate
def scrape_advocate(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='asset-body')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Princeton University
def scrape_princetonuniv(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='field field--name-field-news-body field--type-text-long field--label-hidden field__item')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Milevalue.com
def scrape_milevalue(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='post-content style-light std-block-padding')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Nextcity.org
def scrape_nextcity(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='entry-content')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Dailyreckoning.com
def scrape_dailyreckoning(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='small-12 medium-10 columns single-article')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Decrypt
def scrape_decrypt(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='grid grid-cols-1 md:grid-cols-8 unreset post-content md:pb-20')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for KSAT San Antonio
def scrape_ksatsanantonio(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='dist__Box-sc-1fnzlkn-0 dist__StackBase-sc-1fnzlkn-7 fURyTV iQviKm articleBody')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# scraper for Setopati.com
def scrape_setopati(source, url):
    # create soup object
    soup = create_soup(url)
    
    # if soup is None then return None
    if soup is None:
        print(f'Error for {source} article.\n')
        return None
    
    # isolate article body and paragraphs
    body = soup.find(class_='editor-box')
    paragraphs = [para.text for para in body.find_all('p')]
    
    # populate dictionary
    return populate_content(source, url, paragraphs)

# function to pair source with scraper
def create_source_scraper():
    '''
    How to Use:
        # call scraper
        source_scraper = create_source_scraper()
        
        # run scraper
        scraped_article = source_scraper[source](source, url)
    '''
    # source scraper dictionary
    source_scraper = {'Forbes': scrape_forbes,
                      'Yahoo Entertainment': scrape_yahoo,
                      'Reason': scrape_reason,
                      'Washington Monthly':scrape_washingtonmonthly,
                      'New York Post': scrape_nypost,
                      'GlobeNewswire': scrape_globenewswire,
                      'Investing.com': scrape_investing,
                      'Fox News': scrape_foxnews,
                      'Newsbreak.com': scrape_newsbreak,
                      'CNET': scrape_cnet,
                      'Time': scrape_time,
                      'NPR': scrape_npr,
                      'Al Jazeera English': scrape_aljazeeraenglish,
                      'The New Republic': scrape_newrepublic,
                      'Americanthinker.com': scrape_americanthinker,
                      'CBS News': scrape_cbs,
                      'USA Today': scrape_usatoday,
                      'Daily Signal': scrape_dailysignal,
                      'Breitbart News': scrape_breitbart,
                      'The Week Magazine': scrape_theweekmag,
                      'Boston Herald': scrape_bostonherald,
                      'CNBC': scrape_cnbc,
                      'NerdWallet': scrape_nerdwallet,
                      'Wnd.com': scrape_wnd,
                      'Washington Free Beacon': scrape_washfreebeacon,
                      'ABC News': scrape_abc,
                      'HuffPost': scrape_huffpost,
                      'The Star Online': scrape_thestar,
                      'Abajournal.com': scrape_abajournal,
                      'Roanoke Times': scrape_roanoketimes,
                      'Mediaite': scrape_mediaite,
                      'Westernjournal.com': scrape_westernjournal,
                      'The Root': scrape_theroot,
                      'The Daily Caller': scrape_dailycaller,
                      'ARLnow': scrape_arlnow,
                      'Business Today': scrape_businesstoday,
                      'BusinessLine': scrape_businessline,
                      'Parkersburg News': scrape_parkersburg,
                      'Richmond.com': scrape_richmond,
                      'Thenation.com': scrape_thenation,
                      'ProPublica': scrape_propublica,
                      'blogTO': scrape_blogto,
                      'Lifesciencesworld.com': scrape_lifesciencesworld,
                      'The Denver Post': scrape_denverpost,
                      'LADbible': scrape_ladbible,
                      'Nakedcapitalism.com': scrape_nakedcapitalism,
                      'Catholicnewsagency.com': scrape_catholicnewsagency,
                      'mindbodygreen.com': scrape_mindbodygreen,
                      'The Punch': scrape_thepunch,
                      'Patheos': scrape_patheos,
                      'Elearningindustry.com': scrape_elearningindustry,
                      'Vox': scrape_vox,
                      'The Cut': scrape_thecut,
                      'The Portland Mercury': scrape_portlandmercury,
                      'Slate Magazine': scrape_slatemag,
                      'pymnts.com': scrape_pymnts,
                      'POPSUGAR': scrape_popsugar,
                      'Livemint': scrape_livemint,
                      'Rolling Out': scrape_rollingout,
                      'CNN': scrape_cnn,
                      'Twistedsifter.com': scrape_twistedsifter,
                      'Askamanager.org': scrape_askamanager,
                      'New York Magazine': scrape_newyorkmag,
                      'Raw Story': scrape_rawstory,
                      'Pajiba.com': scrape_pajiba,
                      'Common Dreams': scrape_commondreams,
                      'Hollywood Life': scrape_hollywoodlife,
                      'Chicago Reader': scrape_chicagoreader,
                      'Pmcarpenter.com': scrape_pmcarpenter,
                      'WCVB Boston': scrape_wcvbboston,
                      'WSOC Charlotte': scrape_wsoccharlotte,
                      'WSB Atlanta': scrape_wsbatlanta,
                      'fox6now.com': scrape_fox6now,
                      'The Advocate': scrape_advocate,
                      'Princeton University': scrape_princetonuniv,
                      'Milevalue.com': scrape_milevalue,
                      'Nextcity.org': scrape_nextcity,
                      'Dailyreckoning.com': scrape_dailyreckoning,
                      'Decrypt': scrape_decrypt,
                      'KSAT San Antonio': scrape_ksatsanantonio,
                      'Setopati.com': scrape_setopati}
    
    return source_scraper

# function to check for new sources (not in source_scraper or ineligible)
def check_sources(news_df):
    '''
    Given pandas dataframe NewsAPI extraction, checks if any sources do not have scrapers or aren't yet ineligible.

    Parameters
    ----------
    news_df : pandas dataframe
        NewsAPI extraction.

    Returns
    -------
    list
        List of any sources which aren't yet reviewed.
    '''
    
    # get keys (i.e. current eligible sources)
    eligible = list(create_source_scraper().keys())
    
    # ineligible sources
    ineligible = create_ineligible()
    
    # reviewed sources
    reviewed = eligible + ineligible
    
    # initiate list for potentially new scraping sources
    potential = []
    
    # iterate through news article dataframe and compare to current eligible sources
    sources = news_df['source'].tolist()
    for source in sources:
        if source not in reviewed:
            potential.append(source)
            
    return list(set(potential))

# function to scrape and concatenate
def scrape_and_concatenate(news_df):
    '''
    Scrapes the actual articles associated with the NewsAPI extraction (extraction doesn't give full content).

    Parameters
    ----------
    news_df : pandas dataframe
        NewsAPI extraction.

    Returns
    -------
    scraped_articles : pandas dataframe
        Paragraphs from body of articles associated with NewsAPI extraction.
    '''
    
    # initiate dataframe template
    scraped_articles = pd.DataFrame(columns=['source', 'url', 'paragraph', 'paragraph_num'])
    
    # create source scaper
    source_scraper = create_source_scraper()
    
    # article rows
    num_articles = news_df.shape[0]
    
    # scraped article count
    num_scraped = 0
    
    # iterate through the sources
    for index, row in news_df.iterrows():
        # get source and url
        source = row['source']
        url = row['url']
        
        # call scraper function
        if source in source_scraper.keys():
            try:
                scraped_article = source_scraper[source](source, url)
            except:
                scraped_article = None
            if scraped_article is None:
                continue
            else:
                scraped_articles = pd.concat([scraped_articles, scraped_article], ignore_index=True)
        else:
            print(f'{source} not currently reviewed.\n')
            
        # progress report
        num_scraped += 1
        progress = (num_scraped / num_articles) * 100
        print(f'{progress:.2f}%')
    
    return scraped_articles
