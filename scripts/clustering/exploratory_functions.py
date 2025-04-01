'''
Exploratory Functions
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation
import os
from sklearn.metrics import silhouette_score
import re
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import csv

# import specific functions
from vectorizing_functions import *

# function for additional cleaning of words
def remove_additional_words(text, words):
    for word in words:
        text = text.replace(word, '')
    
    return text

# function to create corpus, takes column(s) as text file name and column for the text data
def create_corpus(df, name_column, content_column, corpus_folder):
    if not os.path.exists(corpus_folder):
        os.makedirs(corpus_folder)
    
    files_to_write = df.shape[0]
    file_progress_checks = [int(interval) for interval in np.linspace(0, files_to_write, 10 + 1)]
    
    for index, row in df.iterrows():
        filename = f'{row[name_column]}.txt'
        filepath = os.path.join(corpus_folder, filename)
        
        # write to file
        with open(filepath, 'w') as file:
            file.write(row[content_column])
            
        # report progress
        if index in file_progress_checks:
            print(f'Progress: {index / files_to_write:.2%}')
            
# function to run lda
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

# function to get n topic words across m topics each - i.e. total = n*m
def return_top_lda(lda_model, num_words):
    # extract the weighted frequencies (pseudocounts)
    lda_df = create_lda_frequencies(lda_model)
    
    # topics
    topics = [col for col in lda_df.columns if col.startswith('topic_')]
    
    # initialize data structures - dictionary to distinguish between topics
    top_words_dict = {topic: None for topic in topics}
    
    # initialize data structures - set to update with unique words
    top_words_set = set()
    
    # cycle through the topics until desired word count is reached
    for num, topic in enumerate(topics):
        # initialize number of words for nlargest()
        n = num_words
        
        # initialize list for topic specific storage (resets on iteration)
        topic_words = []
        
        # while loop until desired word count
        while len(topic_words) < num_words:
            new_words = lda_df[['vocab', topic]].nlargest(n, topic, 'all')['vocab'].tolist()
            unique_new_words = [word for word in new_words if word not in top_words_set]
            topic_words = unique_new_words[:num_words]
            n += 1
        
        # update the main data structures
        top_words_set.update(topic_words)
        top_words_dict[topic] = topic_words
    
    # change data structures
    top_words_list = list(top_words_set)
    top_words_df = pd.DataFrame(top_words_dict)
    
    # return data
    return top_words_list, top_words_df

# function to create list from cv - provides text after cv cleaning
def list_greater_than_zero(cv_df):
    # initialize total collection storage - end length will be total number of documents in cv
    total_collection = []
    
    # iterate through cv
    for index, row in cv_df.iterrows():
        # initialize sub collection storage - will contain the words within the document and cv maximum features
        sub_collection = []
        # iterate through the columns (words) from the cv maximum features
        for col in cv_df.columns:
            # col value within row greater than 0 indicates word is in document
            if row[col] > 0:
                sub_collection.append(col)
        # after column iteration, append back to total collection storage
        total_collection.append(' '.join(sub_collection))
    
    # return
    return total_collection

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

# function to visualize 3d results (Clustering, PCA)
def visualize_results_3d(df_results, label_col, legend_title, save_path, opacity=0.7, arrow_size=10):
    '''
    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame containing the PCA space projection along with the original label column(s).
    label_col : str
        String for the column name representing the label of the data.
    legend_title : str
        Title type string for the legend title.
    save_path : str
        The file path to save the plot. Depending on IDE and settings, likely best to view html visual object in browser.
    opacity : float, optional
        Opacity of projected data points. The default is 0.7.
    arrow_size : int, optional
        Axis length for each principal component. The default is 10.

    Returns
    -------
    None.

    '''
    df = df_results.copy()
    df.rename(columns={label_col: legend_title}, inplace=True)
    
    # begin figure
    fig = px.scatter_3d(df, x='component_1', y='component_2', z='component_3', color=legend_title, opacity=opacity)
    
    # arrow options (principal component axes)
    arrows = [
        go.Scatter3d(x=[-arrow_size, arrow_size], y=[0, 0], z=[0, 0], mode='lines+text', line=dict(color='black', width=8), text=['PC1 (-)', 'PC1 (+)'], textposition='top center', showlegend=False),
        go.Scatter3d(x=[0, 0], y=[-arrow_size, arrow_size], z=[0, 0], mode='lines+text', line=dict(color='black', width=8), text=['PC2 (-)', 'PC2 (+)'], textposition='top center', showlegend=False),
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[-arrow_size, arrow_size], mode='lines+text', line=dict(color='black', width=8), text=['PC3 (-)', 'PC3 (+)'], textposition='top center', showlegend=False)
    ]

    # add arrow traces
    fig.add_traces(arrows)
    
    # disable layout panes
    fig.update_layout(scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ))

    # update legend
    fig.update_layout(legend=dict(
        # title_text=legend_title,
        x=0.1,
        y=0.9,
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor='black',
        borderwidth=2
        ))
    
    # write plotly figure to html
    fig.write_html(save_path)

# function to run kmeans clustering and return results in a dataframe
def kmeans_clustering(pca_df, min_clusters, max_clusters):
    results_df = pca_df.copy()
    
    for cluster in range(min_clusters, max_clusters + 1):
        model = KMeans(n_clusters=cluster, random_state=42)
        model_labels = model.fit_predict(pca_df)
        results_df[f'clusters_{cluster}'] = model_labels
        
    return results_df

# function to create silhouette plot for results
def kmeans_silhouetting(results_df, title, save_path=False):
    # calculate silhouette coefficient scores
    cluster_columns = [col.split('_')[1] for col in results_df.columns if col.startswith('clusters_')]
    silhouette_scores = {'clusters':[], 'scores':[]}
    for cluster in cluster_columns:
        score = silhouette_score(X=results_df[['component_1', 'component_2', 'component_3']], labels=results_df[f'clusters_{cluster}'])
        silhouette_scores['clusters'].append(cluster)
        silhouette_scores['scores'].append(score)
    
    # silhouette dataframe
    silhouette_df = pd.DataFrame(silhouette_scores)
    
    # plot silhouette coefficient scores
    plt.figure()
    sns.lineplot(silhouette_df, x='clusters', y='scores')
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Coefficient Values')
    plt.title(title)
    
    # mark and label the largest 2 scores
    top_scores = silhouette_df.nlargest(2, 'scores')
    for index, row in top_scores.iterrows():
        plt.annotate(f"{row['scores']:.2f}", (row['clusters'], row['scores']), textcoords="offset points", xytext=(0, -20), ha='center', color='black')
        plt.scatter(row['clusters'], row['scores'], color='red')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
        
    return silhouette_df

# function to plot silhouette score averages across different variations of kmeans clustering
def plot_silhouette_averages(silhouette_concatenation, title, groupby_col='Features', col_order=None, save_path=False):
    # groupby features into averages
    silhouette_averages = silhouette_concatenation.groupby(groupby_col).mean().reset_index()
    
    # create plot
    plt.figure()
    if col_order:
        sns.barplot(silhouette_averages, x=groupby_col, y='scores', order=col_order)
    else:
        sns.barplot(silhouette_averages, x=groupby_col, y='scores')
    plt.xlabel(groupby_col)
    plt.ylabel('Average Silhouette Coefficient Score')
    plt.title(title)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()  

# see if sources were given the same clustering
def proportion_clusters_by_label(df, label_col, cluster_col):
    proportion_df = df.groupby([label_col, cluster_col]).size().groupby(level=0).apply(lambda x: x / x.sum()).reset_index(name='proportion')
    return proportion_df

# now we want to visualze the cluster majorities
def calculate_cluster_majorities(proportion_df, label_col):
    majority_cluster_df = proportion_df.loc[proportion_df.groupby(label_col)['proportion'].idxmax()].reset_index(drop=True)
    return majority_cluster_df

# extract reddit post_id
def extract_reddit_post_id(text):
    return text.split('_')[0]

# extract reddit author
def extract_reddit_author(text):
    pattern = r'\d+_(.*?)\.txt'
    pattern_match = re.search(pattern, text)
    return pattern_match.group(1)

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

# function for initial clustering setup
def explore_clusters(text_data, vectorizer_type, pca_save_paths, kmeans_save_paths):
    '''
    Parameters
    ----------
    text_data : TYPE
        DESCRIPTION.
    vectorizer_type : TYPE
        DESCRIPTION.
    pca_save_paths : dictionary
        Dictionary with keys 'max', 'tenth', 'lda_3', and 'lda_5' and values as save paths.
    kmeans_save_paths : dictionary
        Dictionary with keys 'max', 'tenth', 'lda_3', 'lda_5', and 'average' and values as save paths.

    Returns
    -------
    None.

    '''
    
    ## STEP 1 - PERFORM VECTORIZING ##
    # maximum features
    params_max = {'stop_words': 'english'}
    cv_max = vectorize_to_df(text_data, input_type='content', vectorizer_type=vectorizer_type, params=params_max)
    
    # 10% maximum features
    params_tenth = {'stop_words': 'english',
                    'max_features': int(cv_max.shape[1] / 10)}
    cv_tenth = vectorize_to_df(text_data, input_type='content', vectorizer_type=vectorizer_type, params=params_tenth)
    
    # lda 3 topics
    lda_model_3, lda_object_3 = run_lda(df=cv_tenth, num_topics=3, iterations=100, learning='online')
    top_words_list_3, top_words_fd_3 = return_top_lda(lda_model_3, num_words=50)
    cv_lda_3 = cv_tenth[top_words_list_3]
    
    # lda 5 topics
    lda_model_5, lda_object_5 = run_lda(df=cv_tenth, num_topics=5, iterations=100, learning='online')
    top_words_list_5, top_words_fd_5 = return_top_lda(lda_model_5, num_words=30)
    cv_lda_5 = cv_tenth[top_words_list_5]
    
    
    ## STEP 2 - PCA ##
    # run pca
    pca_max = run_pca(cv_max, 'Maximum Features', n_components=3, save_path=pca_save_paths['max'])
    pca_tenth = run_pca(cv_tenth, 'Tenth of Maximum Features', n_components=3, save_path=pca_save_paths['tenth'])
    pca_lda_3 = run_pca(cv_lda_3, 'LDA 3 Topics Features', n_components=3, save_path=pca_save_paths['lda_3'])
    pca_lda_5 = run_pca(cv_lda_5, 'LDA 5 Topics Features', n_components=3, save_path=pca_save_paths['lda_5'])
    
    # save pca results
    pca_max.to_csv(f'{pca_save_paths["max"].replace(".png", ".csv")}', index=False)
    pca_tenth.to_csv(f'{pca_save_paths["tenth"].replace(".png", ".csv")}', index=False)
    pca_lda_3.to_csv(f'{pca_save_paths["lda_3"].replace(".png", ".csv")}', index=False)
    pca_lda_5.to_csv(f'{pca_save_paths["lda_5"].replace(".png", ".csv")}', index=False)
    
    
    ## STEP 3 - KMEANS CLUSTERING ##
    # run kmeans clustering
    kmeans_max = kmeans_clustering(pca_max, min_clusters=2, max_clusters=6)
    kmeans_tenth = kmeans_clustering(pca_tenth, min_clusters=2, max_clusters=6)
    kmeans_lda_3 = kmeans_clustering(pca_lda_3, min_clusters=2, max_clusters=6)
    kmeans_lda_5 = kmeans_clustering(pca_lda_5, min_clusters=2, max_clusters=6)
    
    # save kmeans clustering results
    kmeans_max.to_csv(f'{kmeans_save_paths["max"].replace(".png", ".csv")}', index=False)
    kmeans_tenth.to_csv(f'{kmeans_save_paths["tenth"].replace(".png", ".csv")}', index=False)
    kmeans_lda_3.to_csv(f'{kmeans_save_paths["lda_3"].replace(".png", ".csv")}', index=False)
    pca_lda_5.to_csv(f'{kmeans_save_paths["lda_5"].replace(".png", ".csv")}', index=False)
    
    
    ## STEP 4 - KMEANS SILHOUETTE SCORES ##
    silhouette_max = kmeans_silhouetting(kmeans_max, 'Maximum Features', save_path=kmeans_save_paths['max'])
    silhouette_tenth = kmeans_silhouetting(kmeans_tenth, 'Tenth of Maximum Features', save_path=kmeans_save_paths['tenth'])
    silhouette_lda_3 = kmeans_silhouetting(kmeans_lda_3, 'LDA 3 Topics Features', save_path=kmeans_save_paths['lda_3'])
    silhouette_lda_5 = kmeans_silhouetting(kmeans_lda_5, 'LDA 5 Topics Features', save_path=kmeans_save_paths['lda_5'])
    
    
    ## STEP 5 - KMEANS AVERAGE SILHOUETTE SCORES ##
    # create feature column across the silhouette score results
    silhouette_max['Features'] = 'Maximum'
    silhouette_tenth['Features'] = 'Tenth'
    silhouette_lda_3['Features'] = '3 Topics'
    silhouette_lda_5['Features'] = '5 Topics'
    
    # concatenate silhouette score results
    silhouette_concatenated = pd.concat([silhouette_max, silhouette_tenth], ignore_index=True)
    silhouette_concatenated = pd.concat([silhouette_concatenated, silhouette_lda_3], ignore_index=True)
    silhouette_concatenated = pd.concat([silhouette_concatenated, silhouette_lda_5], ignore_index=True)
    
    # plot averages
    silhouette_col_order = ['Maximum', 'Tenth', '3 Topics', '5 Topics']
    plot_silhouette_averages(silhouette_concatenation=silhouette_concatenated, title='Average Silhouette Scores', groupby_col='Features', col_order=silhouette_col_order, save_path=kmeans_save_paths['average'])
    
    # return values - cv results
    cv_dictionary = {'max': cv_max,
                     'tenth': cv_tenth,
                     'lda_3': cv_lda_3,
                     'lda_5': cv_lda_5}
    
    # return values - kmeans results
    kmeans_dictionary = {'max': kmeans_max,
                         'tenth': kmeans_tenth,
                         'lda_3': kmeans_lda_3,
                         'lda_5': kmeans_lda_5}
    
    return cv_dictionary, kmeans_dictionary

# function to illustrate balance of the clustering results
def illustrate_cluster_balance(kmeans_df, cluster, title, save_path=None, fontsize=15):
    # figure creation
    plt.figure(figsize=(12, 8))
    sns.countplot(kmeans_df, x=cluster)
    plt.xlabel('Cluster', fontsize=fontsize)
    plt.ylabel('Allocation Count', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    
    # if save path specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # show figure
    plt.show()
    
# function to retain a list of provided words in a content column
def retain_words(content, retainable_words):
    return ' '.join([word for word in content.split() if word in retainable_words])

# function to create naming map
def create_naming_map(df, iterative_feature, unique_feature):
    '''
    Parameters
    ----------
    df : pandas dataframe
        Dataframe to create naming map for.
    iterative_feature : string
        Column name in dataframe to create iterative ids for.
    unique_feature : string
        Column name in dataframe which is unique across every row (similary to primary key).

    Returns
    -------
    pandas dataframe
        Naming map with columns:
            - iterative_feature
            - iterative_feature_id
            - unique_feature
            
    Example
    -------
    df:
        - iterative_feature column: news source name (i.e. articles from several repeatable news sources - column contains duplicates)
        - unique_feature column: url to article (unique - column doesn't contain duplicates)
    '''
    
    # initialize source map dictionary
    naming_map = {iterative_feature: [], f'{iterative_feature}_id': [], unique_feature: []}
    
    # iterate through dataframe to obtain information and populate source map
    for index, row in df.iterrows():
        # obtain current features
        row_iterative_feature = row[iterative_feature]
        row_unique_feature = row[unique_feature]
        
        # create source id based on source count
        iterative_feature_count = str(naming_map[iterative_feature].count(row_iterative_feature))
        iterative_feature_id = f'{row_iterative_feature}_{iterative_feature_count}'
        
        # update mapping dictionary
        naming_map[iterative_feature].append(row_iterative_feature)
        naming_map[f'{iterative_feature}_id'].append(iterative_feature_id)
        naming_map[unique_feature].append(row_unique_feature)
        
    # return map as dataframe
    return pd.DataFrame(naming_map)

# function to turn vectorized dataframe into transaction data
def vectorized_to_transaction(vectorized_df, save_path):
    # create collection for each row in the vectorized dataframe
    collection_data = list_greater_than_zero(vectorized_df)
    
    # turn each collection record into a list
    transaction_data = [content.split() for content in collection_data]
    
    # export the transaction data
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for transaction in transaction_data:
            writer.writerow(transaction)
            
    print(f'Transaction Data Successfully Exported: {save_path}')

# function to turn dataframe column into transaction data
def column_to_transaction(df, column, save_path):
    # create transaction data with each record
    transaction_data = []
    for index, row in df.iterrows():
        record_transaction = list(set(row[column].split()))
        transaction_data.append(record_transaction)
        
    # export the transaction data
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for transaction in transaction_data:
            writer.writerow(transaction)
            
    print(f'Transaction Data Successfully Exported: {save_path}')

# function to calculate multiple majorities
def calculate_multiple_majorities(hierarchical_df, label_col):
    # cluster columns
    cluster_columns = [col for col in hierarchical_df.columns if col.startswith('cluster_')]
    
    # iterate through the cluster columns
    for col_num, col in enumerate(cluster_columns):
        # calculate proportions
        proportion_df = proportion_clusters_by_label(hierarchical_df, label_col, col)
        
        # calculate majorities
        iter_majority_df = calculate_cluster_majorities(proportion_df, label_col)
        
        # rename proportions with number of clusters
        clusters = col.split('_')[1]
        new_proportion = f'proportion_{clusters}'
        iter_majority_df.rename(columns={'proportion': new_proportion}, inplace=True)
        
        # create copy of majority_df if first iteration
        if col_num == 0:
            majority_df = iter_majority_df.copy()
        else:
            majority_df = pd.concat([majority_df, iter_majority_df[[col, new_proportion]]], axis=1)
            
    # return majority dataframe
    return majority_df


