'''
modeling functions
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unidecode
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score)

# function to perform minimal cleaning for token preparation
def minimal_cleaning(text):
    # lowercase
    text = text.lower()
    
    # emoji handling
    text = emoji.replace_emoji(text, '')
    
    # accent handling
    text = unidecode.unidecode(text)
    
    # remove website links
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    return text

# contraction expansions
contractions_dict = {"don't": "do not",
                     "didn't": "did not",
                     "isn't": "is not",
                     "wasn't": "was not",
                     "aren't": "are not",
                     "weren't": "were not",
                     "hasn't": "has not",
                     "haven't": "have not",
                     "hadn't": "had not",
                     "can't": "cannot",
                     "couldn't": "could not",
                     "shan't": "shall not",
                     "shouldn't": "should not",
                     "won't": "will not",
                     "wouldn't": "would not",
                     "mightn't": "might not",
                     "mustn't": "must not",
                     "oughtn't": "ought not",
                     "needn't": "need not",
                     "could've": "could have",
                     "should've": "should have",
                     "would've": "would have",
                     "might've": "might have",
                     "must've": "must have",
                     "i'm": "i am",
                     "you're": "you are",
                     "she's": "she is",
                     "he's": "he is",
                     "it's": "it is",
                     "we're": "we are",
                     "they're": "they are",
                     "i've": "i have",
                     "you've": "you have",
                     "we've": "we have",
                     "they've": "they have",
                     "i'll": "i will",
                     "you'll": "you will",
                     "he'll": "he will",
                     "she'll": "she will",
                     "it'll": "it will",
                     "we'll": "we will",
                     "they'll": "they will",
                     "i'd": "i had",
                     "you'd": "you had",
                     "she'd": "she had",
                     "he'd": "he had",
                     "it'd": "it had",
                     "we'd": "we had",
                     "they'd": "they had",
                     "that's": "that is",
                     "that've": "that have",
                     "that'd": "that would",
                     "which've": "which have",
                     "who's": "who is",
                     "who're": "who are",
                     "who've": "who have",
                     "who'd": "who had",
                     "who'll": "who will",
                     "what's": "what is",
                     "what're": "what are",
                     "what'll": "what will",
                     "where's": "where is",
                     "where'd": "where did",
                     "when's": "when is",
                     "why's": "why is",
                     "why'd": "why did",
                     "how's": "how is",
                     "here's": "here is",
                     "there's": "there is",
                     "there'll": "there will",
                     "there'd": "there had",
                     "someone's": "someone is",
                     "somebody's": "somebody is",
                     "no one's": "no one is",
                     "nobody's": "nobody is",
                     "something's": "something is",
                     "nothing's": "nothing is",
                     "let's": "let us",
                     "ma'am": "madam",
                     "o'clock": "of the clock"}


# function to clean text
def clean_text(text, contractions_dict=contractions_dict):
    # lowercase
    text = text.lower()
    
    # remove line breaks
    text = text.replace('\n', ' ')
    
    # remove line spaces
    text = text.replace('\xa0', ' ')
    
    # remove website links
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # ensure no acccent characters
    text = unidecode.unidecode(text)
    
    # ensure no emojis
    text = emoji.replace_emoji(text, ' ')
    
    # apply contraction expansions
    tokens = word_tokenize(text)
    expanded_tokens = [contractions_dict[token] if token in contractions_dict else token for token in tokens]
    text = ' '.join(expanded_tokens)
    
    # remove punctuation and numbers
    text = re.sub(r'[^\w\s]|_', ' ', text)
    
    # remove words containing numbers
    text = re.sub(r'\w*\d\w*', ' ', text)
    
    # remove standalone letters
    text = re.sub(r'\b\w\b', '', text)
    
    # remove leading, trailing, and multi-spaces
    text = text.strip()
    tokens = text.split()
    text = ' '.join(tokens)
    
    return text

# function to remove stopwords
def remove_stopwords(text):
    # nltk stopwords
    stop_words = set(ENGLISH_STOP_WORDS)
    
    # tokenize cleaned text (use split since text will be cleaned prior)
    tokens = text.split()
    
    # apply lemmatizer
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # recombine lemmatized text
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text
    
# function to lemmatize text
def lemmatize_text(text):
    # instantiate lemmer
    lemmer = WordNetLemmatizer()
    
    # tokenize cleaned text (use split since text will be cleaned prior)
    tokens = text.split()
    
    # apply lemmatizer
    lemmatized_tokens = [lemmer.lemmatize(token) for token in tokens]
    
    # recombine lemmatized text
    lemmatized_text = ' '.join(lemmatized_tokens)
    
    return lemmatized_text

# function to remove specific words
def remove_specific(text, specific=['ve']):
    # tokenize
    tokens = text.split()
    
    # remove specific
    clean_tokens = [token for token in tokens if token not in specific]
    
    # recombine clean tokens
    clean_text = ' '.join(clean_tokens)
    
    return clean_text

# function for vectorizer preprocessor
def preprocess(text):
    # clean text
    text = clean_text(text)
    
    # remove stopwords
    text = remove_stopwords(text)
    
    # lemmatize text
    text = lemmatize_text(text)
    
    # remove further specified text
    text = remove_specific(text, specific=['ve', 'll', 'don', 're'])
    
    return text

# function to vectorize dataframe with options
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
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(**params)
    else:
        print("vectorizer_type must be 'count' or 'tfidf'")
        return None
    vectorized = vectorizer.fit_transform(text_data)
    vectorized_columns = vectorizer.get_feature_names_out()
    df = pd.DataFrame(vectorized.toarray(), columns=vectorized_columns)
    
    return df

# function for scaling and pca
def run_pca(cv_df, title, n_components=3, save_path=False):
    scaler = StandardScaler()
    cv_scaled = scaler.fit_transform(cv_df)
    
    pca = PCA(n_components=n_components)
    pca_object = pca.fit_transform(cv_scaled)
    pca_columns = [f'component_{comp + 1}' for comp in range(n_components)]
    pca_df = pd.DataFrame(pca_object, columns=pca_columns)

    # plotting the explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    df_explained = pd.DataFrame({
        'principal_components': [f'principal_component_{col+1}' for col in range(len(explained_variance))],
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance})

    # initialize figure
    plt.figure(figsize=(12, 8))
    # base color blue
    base_color = sns.color_palette()[0]
    # barplot for explained variance by principal component
    sns.barplot(data=df_explained, y='explained_variance', x='principal_components', color=base_color, label='Explained Variance')
    # lineplot for cumulative explained variance by principal component
    sns.lineplot(data=df_explained, y='cumulative_variance', x='principal_components', color='red', label='Cumulative Variance')
    # additional touches
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance')
    plt.xticks(range(df_explained.shape[0]), range(1, df_explained.shape[0] + 1))
    plt.legend(loc='center right')
    plt.grid(True)
    plt.title(f'{title} - {cumulative_variance[-1]:.2%} Explained Variance')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca_df

# function to create confusion matrix
def illustrate_cm(accuracy, cm, model_classes, label_map=None, classes_order=None, title=None, save_path=None):
    # apply label mapping if provided (i.e., {0: 'Right', 1: 'Left})
    if label_map:
        model_classes = [label_map[label] for label in model_classes]
    
    # apply custom order if classes_order not None
    if classes_order:
        ordered_classes = [model_classes.index(label) for label in classes_order]
        ordered_cm = cm[ordered_classes, :][:, ordered_classes]
        disp = ConfusionMatrixDisplay(confusion_matrix=ordered_cm, display_labels=classes_order)
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_classes)
    
    # plotting configuration
    plt.figure(figsize=(16, 12))
    disp.plot()
    plt.xticks(rotation=45, ha='right')
    if title:
        plt.title(f'Accuracy: {accuracy:.2%} for {title}')
    else:
        plt.title(f'Accuracy: {accuracy:.2%}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
