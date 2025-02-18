'''
Vectorizing Functions
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

# function for vectorizing specific cleaning
def specific_cleaning(article):
    # remove line breaks
    article = article.replace('\n', '')
    
    # remove punctuation
    article_letters = re.findall(r'\b[a-zA-Z]+\b', article)
    article = ' '.join(article_letters)
    
    # remove words containing numbers
    article = re.sub(r'\w*\d\w*', '', article)
    
    # remove standalone numbers
    article = re.sub(r'\d+', '', article)
    
    # strip
    article = article.strip()
    
    # lowercase
    article = article.lower()
    
    # remove single length words
    article_tokens = word_tokenize(article)
    filtered_tokens = [token for token in article_tokens if len(token) > 1]
    article = ' '.join(filtered_tokens)
    
    return article
    
# function to remove stop words
def remove_stop_words(cleaned_article):
    # nltk stop words
    stop_words = set(stopwords.words('english'))
    
    # tokenize cleaned article
    cleaned_article_tokens = word_tokenize(cleaned_article)
    
    # remove stopwords from cleaned article tokens
    filtered_article = [token for token in cleaned_article_tokens if token not in stop_words]
    
    # recombine into single string
    article = ' '.join(filtered_article)
    
    return article

# function to lemmatize article - note this does not account for parts of speech (pos), i.e., nouns, adjectives, etc.
def lemmatize_article(cleaned_article):
    # instantiate lemmer
    lemmer = WordNetLemmatizer()
    
    # tokenize cleaned article
    cleaned_article_tokens = word_tokenize(cleaned_article)
    
    # apply lemmatizer
    lemmatized_tokens = [lemmer.lemmatize(token) for token in cleaned_article_tokens]
    
    # recombine into single string
    article_lemmatized = ' '.join(lemmatized_tokens)
    
    return article_lemmatized

# function to stem article - note this does not account for parts of speech (pos), i.e., nouns, adjectives, etc.
def stem_article(cleaned_article):
    # instantiate lemmer
    stemmer = PorterStemmer()
    
    # tokenize cleaned article
    cleaned_article_tokens = word_tokenize(cleaned_article)
    
    # apply lemmatizer
    stemmatized_tokens = [stemmer.stem(token) for token in cleaned_article_tokens]
    
    # recombine into single string
    article_stemmatized = ' '.join(stemmatized_tokens)
    
    return article_stemmatized
    
# function to create word cloud
def create_word_cloud(content, width=1200, height=800, max_words=200, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate'):
    '''
    Notes:
        - generate() and generate_from_text() are synonymous
        - generate_from_frequencies() takes a word:count dictionary, and ignores stopwords
        - cloud_method: choose between 'generate' and 'frequency'
    '''
    # check cloud_method eligibility
    if cloud_method == 'generate':
        wc = WordCloud(width=width,
                       height=height,
                       max_words=max_words,
                       colormap=colormap,
                       min_word_length=min_word_length,
                       background_color=background_color).generate(content)
    elif cloud_method == 'frequency':
        wc = WordCloud(width=width,
                       height=height,
                       max_words=max_words,
                       colormap=colormap,
                       min_word_length=min_word_length,
                       background_color=background_color).generate_from_frequencies(content)
    else:
        print("cloud_method must be 'generate' or 'frequency'")
        return None
        
    return wc

# function to get corpus pathways
def create_corpus_pathways(corpus_path):
    # get the corpus directory
    corpus_files = os.listdir(corpus_path)
    
    # create the corpus pathways
    corpus_pathways = [os.path.join(corpus_path, file) for file in corpus_files]
    
    return corpus_pathways

# function for vectorizing to dataframe with content
def vectorize_to_df(text_data, input_type='content', vectorizer_type='count', params={}):
    '''
    text_data: list
        - for input_type of 'content': list of string values
        - for input_type of 'filename': list of pathways to .txt files in corpus (create with create_corpus_pathways)
    '''
    # add input_type to params
    if input_type == 'content':
        params['input'] = input_type
    elif input_type == 'filename':
        params['input'] = input_type
    else:
        print("input_type must be 'content' or 'filename'")
        return None
    
    # instantiate vectorizer
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(**params)
    elif vectorizer_type == 'tfid':
        vectorizer = TfidfVectorizer(**params)
    else:
        print("vectorizer_type must be 'count' or 'tfid'")
        return None
    vectorized = vectorizer.fit_transform(text_data)
    vectorized_columns = vectorizer.get_feature_names_out()
    df = pd.DataFrame(vectorized.toarray(), columns=vectorized_columns)
    
    # frequency dictionary
    # frequency_dict = df.sum(axis=0).to_dict()
    
    return df

# function to truncate text
def truncate_text(df, columns, max_length=200):
    df_truncated = df.copy()
    for index, row in df.iterrows():
        for col in columns:
            raw_text = row[col]
            if type(raw_text) == str:
                if len(raw_text) > max_length:
                    df_truncated.loc[index, col] = f'{raw_text[:max_length]}...'
                    
    return df_truncated
