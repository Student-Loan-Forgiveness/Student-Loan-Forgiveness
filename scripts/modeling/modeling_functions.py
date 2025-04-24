'''
modeling functions
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import unidecode
import emoji
import string
import csv
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn import tree
from sklearn.inspection import permutation_importance
import plotly_express as px
import plotly.graph_objects as go

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


# function to gather text data from multiple corpuses
def create_corpus_paths(corpus_location):
    # create data structure to store locations
    file_locations = []
    
    # create data structure to store labels
    labels = []
    
    # walk the path
    for root, dirs, files in os.walk(corpus_location):
        for file in files:
            if file.endswith('.txt'):
                # file pathway
                file_locations.append(os.path.join(root, file))
                
                # label
                labels.append(file.split('_')[0].upper())

    return file_locations, labels

# function for transforming count vectorized dataframe non-sequential text file
def cv_to_unsequential(cv_df, label_col):
    # create copy to prevent permeating edits
    df = cv_df.copy()
    
    # extract labels
    labels = df[label_col].unique().tolist()
    
    # label counts
    label_counts = {f'{label}': 0 for label in labels}
    
    # non-label columns
    non_labels = [col for col in cv_df.columns if col != label_col]
    
    # initialize total collection storage
    non_sequentials = {}
    
    # iterate through cv
    for index, row in cv_df.iterrows():
        # create sub non sequentials storage
        sub_non_sequentials = []
        
        label = row[label_col]
        label_counts[label] += 1
        
        # iterate through the remaining columns
        for col in non_labels:
            # extend sub non sequentials
            sub_non_sequentials.extend([col] * row[col])
            
        # populate parent non suquentials storage
        non_sequentials[f'{label.lower()}_mop_{label_counts[label]}.txt'] = ' '.join(sub_non_sequentials)

    return non_sequentials

# function to write unsequential strings to corpus
def unsequential_to_corpus(non_sequentials, save_path='Corpuses/Combined'):
    for non_sequential in non_sequentials:
        file_path = os.path.join(save_path, non_sequential)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(non_sequentials[non_sequential])
            print(f'{non_sequential} saved')

# function to create frequencies from countvectorized dataframes
def create_frequencies(cv_df, label_col=None, max_words=200, ignore_words=None):
    # create copy to prevent permeating edits
    df = cv_df.copy()
    
    # data structure for storage
    frequencies = {}
    
    if label_col:
        labels = df[label_col].unique().tolist()
        words = [col for col in df.columns if col != label_col]
        for label in labels:
            df_split = df[df[label_col]==label]
            df_split.reset_index(drop=True, inplace=True)
            split_frequencies = df_split[words].sum()
            if ignore_words:
                split_frequencies = split_frequencies[~split_frequencies.index.isin(ignore_words)]
            split_top = split_frequencies.nlargest(max_words, keep='first').to_dict()
            frequencies[label] = split_top
    else:
        frequencies = df[words].sum()
        if ignore_words:
            frequencies = frequencies[frequencies.index.isin(ignore_words)]
        top = frequencies.nlargest(max_words, keep='first').to_dict()
        frequencies[label] = top

    return frequencies

# function to create transaction list from count vectorized dataframe
def list_greater_than_zero(cv_df, label_col=None):
    # initialize total collection storage - end length will be total number of documents in cv
    total_collection = []
    
    # iterate through cv
    for index, row in cv_df.iterrows():
        # initialize sub collection storage - will contain the words within the document and cv maximum features
        sub_collection = []
        # iterate through the columns (words) from the cv maximum features
        for col in cv_df.columns:
            # account for label
            if (label_col) and (col == label_col):
                sub_collection.append(row[col])
            # col value within row greater than 0 indicates word is in document
            elif row[col] > 0:
                sub_collection.append(col)
        # after column iteration, append back to total collection storage
        total_collection.append(' '.join(sub_collection))
    
    # turn into lists
    transactions = [collection.split() for collection in total_collection]
    
    # return
    return transactions

# function to save transaction list as transaction data type
def save_as_transaction(transactions, save_path):
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for transaction in transactions:
            writer.writerow(transaction)
            
# function to perform lda
def run_lda(df, num_topics, iterations=50, learning='online'):
    lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=iterations, learning_method=learning)
    lda_object = lda_model.fit_transform(df)
    
    return lda_model, lda_object

# function to create lda top list
def create_lda_top_list(lda_model, lda_object, num_words, fontsize_base, save_path=False):
    # number of topics
    num_topics = lda_object.shape[1]
    
    # topic-word distribution (importance or weight of each word in each topic via probability in distribution)
    topic_word_distribution = np.array(lda_model.components_)
    word_topic_distribution = topic_word_distribution.transpose()
    
    # vocabulary array (all features in - use cv_df columns or lda_model.feature_names_in_)
    vocab_array = np.asarray(lda_model.feature_names_in_)
    
    # begin image creation
    plt.figure(figsize=(num_topics * 17, num_words * 2.25))
    
    # iterate through the topics to find associated words
    for topic in range(num_topics):
        plt.subplot(1, num_topics, topic + 1)  # plot numbering starts with 1
        plt.ylim(0, num_words + 0.5)  # stretch the y-axis to accommodate the words
        plt.xticks([])  # remove x-axis markings ('ticks')
        plt.yticks([]) # remove y-axis markings ('ticks')
        plt.title(f'\nTopic #{topic}\n', fontsize = fontsize_base / 1.25)
        top_words_idx = np.argsort(word_topic_distribution[:, topic])[::-1]  # descending order
        top_words_idx = top_words_idx[:num_words]
        top_words = vocab_array[top_words_idx]
        top_words_shares = word_topic_distribution[top_words_idx, topic]
        for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
            plt.text(0.5, num_words - i - 0.5, word, fontsize = fontsize_base / 2, ha='center')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

# function to return just the lda model - can also be inserted into the lda plotting function
def create_lda_frequencies(lda_model):
    # extract weights
    model_components = lda_model.components_.transpose()
    df_components = pd.DataFrame(model_components, columns=[f'topic_{topic}' for topic in range(model_components.shape[1])])
    
    # extract overall vocabulary
    model_vocab = lda_model.feature_names_in_
    df_vocab = pd.DataFrame(model_vocab, columns=['vocab'])
    
    # concatenate dataframes
    df_model = pd.concat([df_vocab, df_components], axis=1)
    
    # calculate overall frequency
    df_model['total_frequency'] = df_model.select_dtypes(include=['number']).sum(axis=1)
    
    return df_model

# function to create lda frequency plots
def create_lda_frequency_plot(lda_model, num_words, save_path=False, subplot_dims=False, max_limits=False, plot_size=(12, 8), text_size=12):
    # extract weights
    model_components = lda_model.components_.transpose()
    df_components = pd.DataFrame(model_components, columns=[f'topic_{topic}' for topic in range(model_components.shape[1])])
    
    # extract overall vocabulary
    model_vocab = lda_model.feature_names_in_
    df_vocab = pd.DataFrame(model_vocab, columns=['vocab'])
    
    # concatenate dataframes
    df_model = pd.concat([df_vocab, df_components], axis=1)
    
    # calculate overall frequency
    df_model['total_frequency'] = df_model.select_dtypes(include=['number']).sum(axis=1)
    
    # max limits
    if max_limits:
        x_max = int(df_model['total_frequency'].max()) + 10
    
    # calculate difference between topic frequency and overall frequency
    topic_columns = [col for col in df_model.columns if col.startswith('topic_')]
    for num, topic in enumerate(topic_columns):
        df_model[f'remaining_{num}'] = df_model['total_frequency'] - df_model[topic]
        
    # create subplots on single image if specified
    if subplot_dims:
        fig, axes = plt.subplots(subplot_dims[0], subplot_dims[1], figsize=plot_size)
        axes = axes.flatten()
        for num, topic in enumerate(topic_columns):
            df_model_subset = df_model[['vocab', topic, f'remaining_{num}']].nlargest(num_words, topic, keep='all')
            df_model_subset.sort_values(by=topic, inplace=True, ignore_index=True)
            df_model_subset.set_index('vocab').plot(kind='barh', stacked=True, color=['red', 'gray'], ax=axes[num])
            # max limits
            if max_limits:
                axes[num].set_xlim([0, x_max])
            axes[num].set_xlabel('Weight', fontsize=text_size)
            axes[num].set_ylabel('Word', fontsize=text_size)
            axes[num].set_title(f'Topic {num} Word Frequency', fontsize=text_size)
            axes[num].legend(labels=['Topic', 'Overall'], loc='upper left', bbox_to_anchor=(1, 1), title='Frequency', fontsize=text_size)
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    # iterate through to create plots
    else:
        for num, topic in enumerate(topic_columns):
            fig, ax = plt.subplots(figsize=plot_size)
            df_model_subset = df_model[['vocab', topic, f'remaining_{num}']].nlargest(num_words, topic, keep='all')
            df_model_subset.sort_values(by=topic, inplace=True, ignore_index=True)
            df_model_subset.set_index('vocab').plot(kind='barh', stacked=True, color=['red', 'gray'], ax=ax)
            # max limits
            if max_limits:
                ax.set_xlim([0, x_max])
            ax.set_xlabel('Weight', fontsize=text_size)
            ax.set_ylabel('Word', fontsize=text_size)
            ax.set_title(f'Topic {num} Word Frequency', fontsize=text_size)
            ax.legend(labels=['Topic', 'Overall'], loc='upper left', bbox_to_anchor=(1, 1), title='Frequency', fontsize=text_size)
            plt.tight_layout()
            if save_path:
                plt.savefig(f'{save_path}_{topic}.png', dpi=300, bbox_inches='tight')
            plt.show()

# function to create confusion matrix
def illustrate_cm(accuracy, cm, model_classes, classes_order=None, title=None, save_path=None):
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
    
# function to illustrate decision tree
def illustrate_dt(model, X_train, accuracy, title=None, save_path=None):
    plt.figure(figsize=(30, 20))
    # ignore warning for never used on tree_plot, it is used for saving
    tree_plot = tree.plot_tree(model,
                               feature_names=X_train.columns.values,
                               class_names=model.classes_,
                               filled=True)
    if title:
        plt.title(f'{title} - Accuracy: {accuracy:.2%}', fontsize=30)
    else:
        plt.title(f'Accuracy: {accuracy:.2%}', fontsize=30)
    if save_path:
        plt.savefig(f'{save_path}.png', dpi=500, bbox_inches='tight')
    plt.show()
    
# function to calculate feature importance via permutation importance
def calculate_feature_importance(model, X_test, y_test):
    permutation_metrics = permutation_importance(model, X_test, y_test)
    features = model.feature_names_in_
    sorted_index = permutation_metrics['importances_mean'].argsort()
    importance_means = permutation_metrics['importances_mean'][sorted_index]
    feature_importance = pd.DataFrame({'feature': features[sorted_index],
                                       'importance': importance_means,
                                       'absolute_importance': abs(importance_means)})
    return feature_importance

# function to aggregate reddit dataframe into a schema
def aggregate_into_schema(reddit_df, retain_columns, aggregate_by):
    df = reddit_df[retain_columns].copy()
    retain_columns.remove(aggregate_by)
    
    # get column types
    column_types = {}
    for col in retain_columns:
        column_types[col] = type(df.loc[0, col])
    
    # aggregate one column at a time in case of list type data
    aggregate_subsets = {}
    for col in retain_columns:
        if column_types[col] == list:
            aggregate_subsets[col] = df.groupby(aggregate_by)[col].apply(lambda x: sum(x, [])).reset_index()
        else:
            aggregate_subsets[col] = df.groupby(aggregate_by)[col].apply(' '.join).reset_index()
            
    # combine datasets
    for subset_num, subset in enumerate(aggregate_subsets):
        if subset_num == 0:
            aggregation_df = aggregate_subsets[subset].copy()
        else:
            aggregation_df = pd.merge(aggregation_df, aggregate_subsets[subset], on=aggregate_by)
    
    return aggregation_df

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

# function to plot overall and training proportions
def plot_overall_training(cv_df, training_data, label_order, label_col='BIAS', title=None, save_path=None):
    total_overall = len(cv_df)
    total_training = len(training_data)
    balance_overall = cv_df[label_col].value_counts(normalize=True).reset_index()
    balance_overall.columns = [label_col, 'Overall']
    balance_training = training_data.value_counts(normalize=True).reset_index()
    balance_training.columns = [label_col, 'Training']
    balance = pd.merge(balance_overall, balance_training, on=label_col)
    balance_melted = balance.melt(id_vars=label_col)
    sns.barplot(balance_melted, x='value', y=label_col, hue='variable',
                order=label_order,
                hue_order=['Overall', 'Training'])
    plt.xlabel('Proportion by Set')
    plt.ylabel('Political Bias')
    if title:
        plt.title(f'Balance of {title} Labels\nOverall Counts: {total_overall} - Training Counts: {total_training}')
    else:
        plt.title(f'Balance of Labels\nOverall Counts: {total_overall} - Training Counts: {total_training}')
    plt.legend(title='Set', loc='upper left', bbox_to_anchor=(1, 1))
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# function to populate and subset to important features
def populate_important_features(cv_df, important_features):
    # create copy of dataframe to prevent permeating edits
    df = cv_df.copy()
    
    # get current features
    current_features = df.columns.tolist()

    # account for missing features
    missing_features = [col for col in important_features if col not in current_features]

    # populate missing features with zeros
    for feature in missing_features:
        df[feature] = 0
        
    # subset features on important features
    df = df[important_features]
    
    # return subset
    return df

# function to iterate through data with a grid search object
def iterate_grid_search(model, param_grid, labels, X_data, y_data, cv=5, verbose=3, return_detail=False):
    # initialize list data structures for results
    best_parameters = {}
    best_scores = []
    if return_detail:
        results_detailed = {}
    
    for label_num, label in enumerate(labels):
        # instantiate grid search object
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=param_grid,
                                   cv=cv,
                                   n_jobs=-1,
                                   verbose=verbose,
                                   scoring='accuracy',
                                   return_train_score=True)
        
        # fit grid search object
        grid_search.fit(X_data[label_num], y_data[label_num])
        
        # retrieve best parameters and scores
        best_parameters_set = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # populate data structures for storage
        best_scores.append(best_score)
        
        if label_num == 0:
            for parameter in best_parameters_set:
                best_parameters[parameter.title()] = [best_parameters_set[parameter]]
        else:
            for parameter in best_parameters_set:
                best_parameters[parameter.title()].append(best_parameters_set[parameter])
                
        if return_detail:
            results_detailed[label] = grid_search.cv_results_

        print(f'Grid Search Complete for: {label}\n')
        
    # create dictionary data structure with populated lists
    model_results = best_parameters.copy()
    model_results['Model'] = labels
    model_results['Accuracy'] = best_scores
    
    # create pandas dataframe (except if dimensions do not fit)
    try:
        results_df = pd.DataFrame(model_results)
        if return_detail:
            return results_df, results_detailed
        else:
            return results_df
    except:
        if return_detail:
            return model_results, results_detailed
        else:
            return model_results
        
# function to combine the detailed results
def combine_iterated_details(results_detailed):
    for label_num, detailed_df in enumerate(results_detailed):
        # copy to prevent permeating edits
        new_combine_df = pd.DataFrame(results_detailed[detailed_df]).copy()
        
        # create label column
        new_combine_df['Label'] = detailed_df
        
        # subset to select columns
        new_combine_df = new_combine_df[['Label', 'mean_test_score', 'rank_test_score'] + [col for col in new_combine_df.columns if col.startswith('param_')] + ['mean_fit_time']]
    
        # create or combine together
        if label_num == 0:
            combined_details = new_combine_df.copy()
        else:
            combined_details = pd.concat([combined_details, new_combine_df], ignore_index=True)
            
    return combined_details

# function to create interactive sentiment gauge
def create_sentiment_gauge(reddit_scaled, save_path):
    # initialize figure
    fig = go.Figure()
    # dropdown buttons
    dropdown_buttons = []
    for index, row in reddit_scaled.iterrows():
        # traces
        fig.add_trace(go.Indicator(
            mode='gauge+number',
            value=row['Sentiment Score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f'Author: {row["Author"]}<br>Sentiment: {row["Sentiment Label"]}'},
            gauge={
                'axis': {'range': [-3, 3]},
                'bar': {'color': 'rgba(0, 0, 0, 0)'},
                'steps': [
                    {'range': [-3, -1], 'color': 'red'},
                    {'range': [-1, 1], 'color': 'blue'},
                    {'range': [1, 3], 'color': 'green'}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 5},
                    'thickness': 1,
                    'value': row['Sentiment Score']
                }
            },
            visible=False
        ))
        
        # dropdown menu
        dropdown_buttons.append(dict(
            label=row['Author'],
            method='update',
            args=[{'visible': [author == index for author in range(len(reddit_scaled))]}]
        ))
        
    # set default visible author(s)
    fig.data[0].visible = True
    
    fig.update_layout(updatemenus=[
        dict(
            buttons=dropdown_buttons,
            direction='down',
            showactive=True
        )
    ])
    
    fig.write_html(save_path)
