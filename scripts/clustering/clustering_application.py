'''
Clustering Section Code

Data:
    - NewsAPI:
        - Topic Specific Data: "student loan forgivess" and "student loans" topic API pulls
        - Full Specific: topic specific data with and without political bias labels
        - Labeled Specific: topic specific data with political bias labels
        - Labeled Plus: general data with political bias labels
    - Reddit:
        - Description
        - Aggregation Schema 1: author's content aggregated within a single thread (BASE)
        - Aggregation Schema 2: author's content aggregated over all threads
        - Aggregation Schema 3: all content aggregated by thread
        - Aggregation Schema 4: all content aggregated by Subreddit

KMeans Strategy:
    - NewsAPI using Labeled Specific:
        - Apply additional cleaning for vectorizing
        - CVs (4):
            - Maximum Features
            - Percentage of Features
            - LDA to size of Percentage of Features:
                - 3 Topics
                - 5 Topics
        - Perform 3-Dimensional PCA on these 4 subsets and report results
        - Perform KMeans Clustering on the 3-Dimensional PCA subsets
        - Evaluate KMeans Clustering results with Silhouette Scores:
            - Show Silhouette Scores per subset
            - Show Average Silhouette Scores across the subsets
        - Choose optimal number of clusters per subset and visualize with 3-Dimensional Plotting
    - Reddit Data:
        - Apply additional cleaning for vectorizing
        - Create aggregation schemas
        - Use most successful approach from the NewsAPI series and run over each schema (4) using TfidfVectorizer
'''

# import specific functions
from vectorizing_functions import *
from exploratory_functions import *

## NEWSAPI - SET UP ##
newsapi_labeled = pd.read_csv('../send_files/cleaning/newsapi_data_cleaned/newsapi_labeled_2_14_25.csv')

# reduce to labeled specific
newsapi_labeled.dropna(subset='Bias Specific', inplace=True)

# reset index
newsapi_labeled.reset_index(drop=True, inplace=True)

# apply function - further specific cleaning
newsapi_labeled['cleaned_article'] = newsapi_labeled['article'].apply(specific_cleaning)

# apply function - lemmatize
newsapi_labeled['lemmatized_article'] = newsapi_labeled['cleaned_article'].apply(lemmatize_article)

# remove additional words
newsapi_labeled['lemmatized_article'] = newsapi_labeled['lemmatized_article'].apply(lambda x: remove_additional_words(x, ['wa', 'ha', 'tt', 've', 'wt', 'tn']))

# text data
newsapi_text_data = newsapi_labeled['lemmatized_article'].tolist()

# save paths - pca
newsapi_pca_paths = {'max': 'kmeans_results/pca_results/newsapi_max.png',
                     'tenth': 'kmeans_results/pca_results/newsapi_tenth.png',
                     'lda_3': 'kmeans_results/pca_results/newsapi_lda_3.png',
                     'lda_5': 'kmeans_results/pca_results/newsapi_lda_5.png'}

# save paths - kmeans
newsapi_kmeans_paths = {'max': 'kmeans_results/silhouette_results/newsapi_max.png',
                        'tenth': 'kmeans_results/silhouette_results/newsapi_tenth.png',
                        'lda_3': 'kmeans_results/silhouette_results/newsapi_lda_3.png',
                        'lda_5': 'kmeans_results/silhouette_results/newsapi_lda_5.png',
                        'average': 'kmeans_results/silhouette_results/newsapi_average.png'}

## NEWSAPI - CLUSTERING ##
cv_dictionary_newsapi, kmeans_dictionary_newsapi = explore_clusters(text_data=newsapi_text_data,
                                                                    vectorizer_type='count',
                                                                    pca_save_paths=newsapi_pca_paths,
                                                                    kmeans_save_paths=newsapi_kmeans_paths)

## NEWSAPI - CLUSTER PICK ##
'''
- Max: 3 (or 2)
- Tenth: 2 (or 6)
- LDA 3: 4 (or 3)
- LDA 3: 2 (or 4)
'''
# visualize 3D
visualize_results_3d(kmeans_dictionary_newsapi['max'], 'clusters_3', 'Clusters', 'kmeans_results/newsapi_maximum_3_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_newsapi['tenth'], 'clusters_2', 'Clusters', 'kmeans_results/newsapi_tenth_2_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_newsapi['tenth'], 'clusters_6', 'Clusters', 'kmeans_results/newsapi_tenth_6_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_newsapi['lda_3'], 'clusters_4', 'Clusters', 'kmeans_results/newsapi_lda_3_4_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_newsapi['lda_5'], 'clusters_2', 'Clusters', 'kmeans_results/newsapi_lda_5_2_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_newsapi['lda_5'], 'clusters_4', 'Clusters', 'kmeans_results/newsapi_lda_5_4_clusters.html', opacity=0.7, arrow_size=10)

'''
cluster choice:
    - Iterative 3 Topic LDA with 4 Clusters
    - Highest coefficient value outside the results which have the major cluster as the mass in the origin and then a few points outside as their own clusters
    - Separates the origin mass well
'''

## NEWSAPI - FINAL CLUSTER ILLUSTRATIONS ##

# illustrate cluster balance
illustrate_cluster_balance(kmeans_dictionary_newsapi['lda_3'], 'clusters_4', 'Iterative 3 Topic LDA', save_path='kmeans_results/newsapi_cluster_selection.png')

# concatenate the cluster choice into the main dataframe
newsapi_cluster_selection = kmeans_dictionary_newsapi['lda_3'][['clusters_4']]
newsapi_cluster_selection.rename(columns={'clusters_4': 'Cluster'}, inplace=True)
newsapi_clustered = pd.concat([newsapi_labeled, newsapi_cluster_selection], axis=1)

# group and concatenate content by cluster
newsapi_cluster_grouped = newsapi_clustered.groupby('Cluster')['lemmatized_article'].apply(' '.join).reset_index()

# extract cluster content from grouping
cluster_content_0 = newsapi_cluster_grouped[newsapi_cluster_grouped['Cluster'] == 0]['lemmatized_article'].values[0]
cluster_content_1 = newsapi_cluster_grouped[newsapi_cluster_grouped['Cluster'] == 1]['lemmatized_article'].values[0]
cluster_content_2 = newsapi_cluster_grouped[newsapi_cluster_grouped['Cluster'] == 2]['lemmatized_article'].values[0]
cluster_content_3 = newsapi_cluster_grouped[newsapi_cluster_grouped['Cluster'] == 3]['lemmatized_article'].values[0]

# create word clouds for top 100 words
wc_0 = create_word_cloud(cluster_content_0, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')
wc_1 = create_word_cloud(cluster_content_1, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')
wc_2 = create_word_cloud(cluster_content_2, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')
wc_3 = create_word_cloud(cluster_content_3, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')

# plot the word clouds
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# flatten the axs array for easier indexing
axs = axs.flatten()

# list of word clouds and titles
wordclouds = [wc_0, wc_1, wc_2, wc_3]
titles = [
    'NewsAPI - Cluster 0',
    'NewsAPI - Cluster 1',
    'NewsAPI - Cluster 2',
    'NewsAPI - Cluster 3'
]

# plot each word cloud in the subplot
for i, wc in enumerate(wordclouds):
    axs[i].imshow(wc, interpolation='bilinear')
    axs[i].axis('off')
    axs[i].set_title(titles[i])

# hide any unused subplots
for j in range(len(wordclouds), len(axs)):
    fig.delaxes(axs[j])

# save the combined image
plt.savefig('kmeans_results/wordclouds/newsapi_clusters.png', dpi=500)

# show the combined image
plt.show()


## REDDIT - SET UP ##

# import reddit data
reddit_student_loan_forgiveness = pd.read_csv('../send_files/cleaning/reddit_data_cleaned/labeled_student_loan_forgiveness.csv')
reddit_student_loans = pd.read_csv('../send_files/cleaning/reddit_data_cleaned/labeled_student_loans.csv')

# add labels for search
reddit_student_loan_forgiveness['search'] = 'reddit_student_loan_forgiveness'
reddit_student_loans['search'] = 'reddit_student_loans'

# concatenate
reddit_labeled = pd.concat([reddit_student_loan_forgiveness, reddit_student_loans], ignore_index=True)

# apply function - further specific cleaning
reddit_labeled['cleaned_content'] = reddit_labeled['author_content_aggregated'].apply(specific_cleaning)

# apply function - lemmatize
reddit_labeled['lemmatized_content'] = reddit_labeled['cleaned_content'].apply(lemmatize_article)

# remove additional words
reddit_labeled['lemmatized_content'] = reddit_labeled['lemmatized_content'].apply(lambda x: remove_additional_words(x, ['wa', 'ha', 'tt', 've', 'wt', 'tn']))

## REDDIT - ILLUSTRATE TFIDFVECTORIZER NEED ##

# get sentence length
reddit_labeled['word_count'] = reddit_labeled['lemmatized_content'].apply(lambda x: len(x.split()))

# reduce to content with "effective" communication levels (https://www.thoughtco.com/sentence-length-grammar-and-composition-1691948)
# also removes reactions and single word replies - since the aim isn't to track 
reddit_labeled = reddit_labeled[reddit_labeled['word_count'] >= 15]

# reset index
reddit_labeled.reset_index(drop=True, inplace=True)

# binning test
reddit_labeled['binned_lengths'] = pd.qcut(reddit_labeled['word_count'], q=4)

# examine word counts by quantile - shows why tfidf is a logical choice
plt.figure(figsize=(12, 8))
sns.countplot(reddit_labeled, x='binned_lengths')
plt.xlabel('Word Count Quartile Bin', fontsize=15)
plt.ylabel('Post Count', fontsize=15)
plt.title('Reddit Post Quartile Binning', fontsize=15)
plt.savefig('kmeans_results/reddit_tfidf_reasoning.png', dpi=300, bbox_inches='tight')
plt.show()

# text data
reddit_base_text_data = reddit_labeled['lemmatized_content'].tolist()

# save paths - pca
reddit_base_pca_paths = {'max': 'kmeans_results/pca_results/reddit_base_max.png',
                         'tenth': 'kmeans_results/pca_results/reddit_base_tenth.png',
                         'lda_3': 'kmeans_results/pca_results/reddit_base_lda_3.png',
                         'lda_5': 'kmeans_results/pca_results/reddit_base_lda_5.png'}

# save paths - kmeans
reddit_base_kmeans_paths = {'max': 'kmeans_results/silhouette_results/reddit_base_max.png',
                            'tenth': 'kmeans_results/silhouette_results/reddit_base_tenth.png',
                            'lda_3': 'kmeans_results/silhouette_results/reddit_base_lda_3.png',
                            'lda_5': 'kmeans_results/silhouette_results/reddit_base_lda_5.png',
                            'average': 'kmeans_results/silhouette_results/reddit_base_average.png'}


## REDDIT - BASE - CLUSTERING ##
cv_dictionary_reddit_base, kmeans_dictionary_reddit_base = explore_clusters(text_data=reddit_base_text_data,
                                                                            vectorizer_type='tfid',
                                                                            pca_save_paths=reddit_base_pca_paths,
                                                                            kmeans_save_paths=reddit_base_kmeans_paths)

## REDDIT - BASE - CLUSTER PICK ##
'''
- Max: 2 (or 4)
- Tenth: 2 (or 4)
- LDA 3: 4 (or 5)
- LDA 3: 3 (or 2)
'''
# visualize 3D
visualize_results_3d(kmeans_dictionary_reddit_base['max'], 'clusters_4', 'Clusters', 'kmeans_results/reddit_base_maximum_4_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_base['tenth'], 'clusters_2', 'Clusters', 'kmeans_results/reddit_base_tenth_2_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_base['tenth'], 'clusters_4', 'Clusters', 'kmeans_results/reddit_base_tenth_4_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_base['lda_3'], 'clusters_4', 'Clusters', 'kmeans_results/reddit_base_lda_3_4_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_base['lda_5'], 'clusters_3', 'Clusters', 'kmeans_results/reddit_base_lda_5_3_clusters.html', opacity=0.7, arrow_size=10)

'''
cluster choice:
    - Iterative 5 Topic LDA with 3 Clusters
    - Highest coefficient value outside the results which have the major cluster as the mass in the origin and then a few points outside as their own clusters
    - Separates the origin mass well
'''

## REDDIT - BASE - FINAL CLUSTER ILLUSTRATIONS ##

# illustrate cluster balance
illustrate_cluster_balance(kmeans_dictionary_reddit_base['lda_5'], 'clusters_3', 'Iterative 5 Topic LDA', save_path='kmeans_results/reddit_base_cluster_selection.png')

# concatenate the cluster choice into the main dataframe
reddit_base_cluster_selection = kmeans_dictionary_reddit_base['lda_5'][['clusters_3']]
reddit_base_cluster_selection.rename(columns={'clusters_3': 'Cluster'}, inplace=True)
reddit_base_clustered = pd.concat([reddit_labeled, reddit_base_cluster_selection], axis=1)

# group and concatenate content by cluster
reddit_base_cluster_grouped = reddit_base_clustered.groupby('Cluster')['lemmatized_content'].apply(' '.join).reset_index()

# extract cluster content from grouping
cluster_content_0 = reddit_base_cluster_grouped[reddit_base_cluster_grouped['Cluster'] == 0]['lemmatized_content'].values[0]
cluster_content_1 = reddit_base_cluster_grouped[reddit_base_cluster_grouped['Cluster'] == 1]['lemmatized_content'].values[0]
cluster_content_2 = reddit_base_cluster_grouped[reddit_base_cluster_grouped['Cluster'] == 2]['lemmatized_content'].values[0]

# create word clouds for top 100 words
wc_0 = create_word_cloud(cluster_content_0, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')
wc_1 = create_word_cloud(cluster_content_1, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')
wc_2 = create_word_cloud(cluster_content_2, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')

# plot the word clouds
fig, axs = plt.subplots(3, 1, figsize=(12, 8))

# wordcloud for cluster 0
axs[0].imshow(wc_0, interpolation='bilinear')
axs[0].axis('off')
axs[0].set_title('Reddit Base Schema - Cluster 0')

# wordcloud for cluster 1
axs[1].imshow(wc_1, interpolation='bilinear')
axs[1].axis('off')
axs[1].set_title('Reddit Base Schema - Cluster 1')

# wordcloud for cluster 2
axs[2].imshow(wc_2, interpolation='bilinear')
axs[2].axis('off')
axs[2].set_title('Reddit Base Schema - Cluster 2')

# Save the combined image
plt.savefig('kmeans_results/wordclouds/reddit_base_schema_clusters.png', dpi=500)

# Show the combined image
plt.show()


## REDDIT - CREATE AUTHOR AGGREGATION SCHEMA ##

# retain the following columns during aggregation
retain_columns = ['author', 'lemmatized_content']

# perform aggregation
reddit_author_aggregation = aggregate_into_schema(reddit_labeled, retain_columns, 'author')

# text data
reddit_author_text_data = reddit_author_aggregation['lemmatized_content'].tolist()

# save paths - pca
reddit_author_pca_paths = {'max': 'kmeans_results/pca_results/reddit_author_max.png',
                           'tenth': 'kmeans_results/pca_results/reddit_author_tenth.png',
                           'lda_3': 'kmeans_results/pca_results/reddit_author_lda_3.png',
                           'lda_5': 'kmeans_results/pca_results/reddit_author_lda_5.png'}

# save paths - kmeans
reddit_author_kmeans_paths = {'max': 'kmeans_results/silhouette_results/reddit_author_max.png',
                              'tenth': 'kmeans_results/silhouette_results/reddit_author_tenth.png',
                              'lda_3': 'kmeans_results/silhouette_results/reddit_author_lda_3.png',
                              'lda_5': 'kmeans_results/silhouette_results/reddit_author_lda_5.png',
                              'average': 'kmeans_results/silhouette_results/reddit_author_average.png'}


## REDDIT - AUTHOR - CLUSTERING ##
cv_dictionary_reddit_author, kmeans_dictionary_reddit_author = explore_clusters(text_data=reddit_author_text_data,
                                                                                vectorizer_type='tfid',
                                                                                pca_save_paths=reddit_author_pca_paths,
                                                                                kmeans_save_paths=reddit_author_kmeans_paths)

## REDDIT - AUTHOR - CLUSTER PICK ##
'''
- Max: 2 (or 4)
- Tenth: 3 (or 2)
- LDA 3: 5 (or 4)
- LDA 3: 4 (or 5)
'''
# visualize 3D
visualize_results_3d(kmeans_dictionary_reddit_author['max'], 'clusters_4', 'Clusters', 'kmeans_results/reddit_author_maximum_4_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_author['tenth'], 'clusters_3', 'Clusters', 'kmeans_results/reddit_author_tenth_3_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_author['lda_3'], 'clusters_5', 'Clusters', 'kmeans_results/reddit_author_lda_3_5_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_author['lda_5'], 'clusters_4', 'Clusters', 'kmeans_results/reddit_author_lda_5_4_clusters.html', opacity=0.7, arrow_size=10)

'''
cluster choice:
    - Iterative 5 Topic LDA with 4 Clusters
    - Highest coefficient value outside the results which have the major cluster as the mass in the origin and then a few points outside as their own clusters
    - Separates the origin mass well
'''

## REDDIT - AUTHOR - FINAL CLUSTER ILLUSTRATIONS ##

# illustrate cluster balance
illustrate_cluster_balance(kmeans_dictionary_reddit_author['lda_5'], 'clusters_4', 'Iterative 5 Topic LDA', save_path='kmeans_results/reddit_author_cluster_selection.png')

# concatenate the cluster choice into the main dataframe
reddit_author_cluster_selection = kmeans_dictionary_reddit_author['lda_5'][['clusters_4']]
reddit_author_cluster_selection.rename(columns={'clusters_4': 'Cluster'}, inplace=True)
reddit_author_clustered = pd.concat([reddit_labeled, reddit_author_cluster_selection], axis=1)

# group and concatenate content by cluster
reddit_author_cluster_grouped = reddit_author_clustered.groupby('Cluster')['lemmatized_content'].apply(' '.join).reset_index()

# extract cluster content from grouping
cluster_content_0 = reddit_author_cluster_grouped[reddit_author_cluster_grouped['Cluster'] == 0]['lemmatized_content'].values[0]
cluster_content_1 = reddit_author_cluster_grouped[reddit_author_cluster_grouped['Cluster'] == 1]['lemmatized_content'].values[0]
cluster_content_2 = reddit_author_cluster_grouped[reddit_author_cluster_grouped['Cluster'] == 2]['lemmatized_content'].values[0]
cluster_content_3 = reddit_author_cluster_grouped[reddit_author_cluster_grouped['Cluster'] == 3]['lemmatized_content'].values[0]

# create word clouds for top 100 words
wc_0 = create_word_cloud(cluster_content_0, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')
wc_1 = create_word_cloud(cluster_content_1, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')
wc_2 = create_word_cloud(cluster_content_2, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')
wc_3 = create_word_cloud(cluster_content_3, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')

# plot the word clouds
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# flatten the axs array for easier indexing
axs = axs.flatten()

# list of word clouds and titles
wordclouds = [wc_0, wc_1, wc_2, wc_3]
titles = [
    'Reddit - Author - Cluster 0',
    'Reddit - Author - Cluster 1',
    'Reddit - Author - Cluster 2',
    'Reddit - Author - Cluster 3'
]

# plot each word cloud in the subplot
for i, wc in enumerate(wordclouds):
    axs[i].imshow(wc, interpolation='bilinear')
    axs[i].axis('off')
    axs[i].set_title(titles[i])

# hide any unused subplots
for j in range(len(wordclouds), len(axs)):
    fig.delaxes(axs[j])

# save the combined image
plt.savefig('kmeans_results/wordclouds/reddit_author_schema_clusters.png', dpi=500)

# show the combined image
plt.show()


## REDDIT - CREATE THREAD AGGREGATION SCHEMA ##

# retain the following columns during aggregation
retain_columns = ['url', 'lemmatized_content']

# perform aggregation
reddit_thread_aggregation = aggregate_into_schema(reddit_labeled, retain_columns, 'url')

# create unique titles for threads (represented by url)
reddit_thread_aggregation['thread'] = reddit_thread_aggregation.index.to_series().apply(lambda x: f'thread_{x}')

# text data
reddit_thread_text_data = reddit_thread_aggregation['lemmatized_content'].tolist()

# save paths - pca
reddit_thread_pca_paths = {'max': 'kmeans_results/pca_results/reddit_thread_max.png',
                           'tenth': 'kmeans_results/pca_results/reddit_thread_tenth.png',
                           'lda_3': 'kmeans_results/pca_results/reddit_thread_lda_3.png',
                           'lda_5': 'kmeans_results/pca_results/reddit_thread_lda_5.png'}

# save paths - kmeans
reddit_thread_kmeans_paths = {'max': 'kmeans_results/silhouette_results/reddit_thread_max.png',
                              'tenth': 'kmeans_results/silhouette_results/reddit_thread_tenth.png',
                              'lda_3': 'kmeans_results/silhouette_results/reddit_thread_lda_3.png',
                              'lda_5': 'kmeans_results/silhouette_results/reddit_thread_lda_5.png',
                              'average': 'kmeans_results/silhouette_results/reddit_thread_average.png'}


## REDDIT - THREAD - CLUSTERING ##
cv_dictionary_reddit_thread, kmeans_dictionary_reddit_thread = explore_clusters(text_data=reddit_thread_text_data,
                                                                                vectorizer_type='tfid',
                                                                                pca_save_paths=reddit_thread_pca_paths,
                                                                                kmeans_save_paths=reddit_thread_kmeans_paths)

## REDDIT - THREAD - CLUSTER PICK ##
'''
- Max: 3 (or 4)
- Tenth: 6 (or 4)
- LDA 3: 4 (or 3)
- LDA 3: 5 (or 6)
'''
# visualize 3D
visualize_results_3d(kmeans_dictionary_reddit_thread['max'], 'clusters_3', 'Clusters', 'kmeans_results/reddit_thread_maximum_3_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_thread['tenth'], 'clusters_6', 'Clusters', 'kmeans_results/reddit_thread_tenth_6_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_thread['lda_3'], 'clusters_4', 'Clusters', 'kmeans_results/reddit_thread_lda_3_4_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_thread['lda_5'], 'clusters_5', 'Clusters', 'kmeans_results/reddit_thread_lda_5_5_clusters.html', opacity=0.7, arrow_size=10)

'''
cluster choice:
    - Data is very spread out compared to the others, probably due to substantially less data
    - Iterative 3 Topic LDA with 4 Clusters
    - Highest coefficient value outside the results which have the major cluster as the mass in the origin and then a few points outside as their own clusters
    - Separates the origin mass well
'''

## REDDIT - THREAD - FINAL CLUSTER ILLUSTRATIONS ##

# illustrate cluster balance
illustrate_cluster_balance(kmeans_dictionary_reddit_thread['lda_3'], 'clusters_4', 'Iterative 3 Topic LDA', save_path='kmeans_results/reddit_thread_cluster_selection.png')

# concatenate the cluster choice into the main dataframe
reddit_thread_cluster_selection = kmeans_dictionary_reddit_thread['lda_3'][['clusters_4']]
reddit_thread_cluster_selection.rename(columns={'clusters_4': 'Cluster'}, inplace=True)
reddit_thread_clustered = pd.concat([reddit_labeled, reddit_thread_cluster_selection], axis=1)

# group and concatenate content by cluster
reddit_thread_cluster_grouped = reddit_thread_clustered.groupby('Cluster')['lemmatized_content'].apply(' '.join).reset_index()

# extract cluster content from grouping
cluster_content_0 = reddit_thread_cluster_grouped[reddit_thread_cluster_grouped['Cluster'] == 0]['lemmatized_content'].values[0]
cluster_content_1 = reddit_thread_cluster_grouped[reddit_thread_cluster_grouped['Cluster'] == 1]['lemmatized_content'].values[0]
cluster_content_2 = reddit_thread_cluster_grouped[reddit_thread_cluster_grouped['Cluster'] == 2]['lemmatized_content'].values[0]
cluster_content_3 = reddit_thread_cluster_grouped[reddit_thread_cluster_grouped['Cluster'] == 3]['lemmatized_content'].values[0]

# create word clouds for top 100 words
wc_0 = create_word_cloud(cluster_content_0, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')
wc_1 = create_word_cloud(cluster_content_1, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')
wc_2 = create_word_cloud(cluster_content_2, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')
wc_3 = create_word_cloud(cluster_content_3, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')

# plot the word clouds
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# flatten the axs array for easier indexing
axs = axs.flatten()

# list of word clouds and titles
wordclouds = [wc_0, wc_1, wc_2, wc_3]
titles = [
    'Reddit - Thread - Cluster 0',
    'Reddit - Thread - Cluster 1',
    'Reddit - Thread - Cluster 2',
    'Reddit - Thread - Cluster 3'
]

# plot each word cloud in the subplot
for i, wc in enumerate(wordclouds):
    axs[i].imshow(wc, interpolation='bilinear')
    axs[i].axis('off')
    axs[i].set_title(titles[i])

# hide any unused subplots
for j in range(len(wordclouds), len(axs)):
    fig.delaxes(axs[j])

# save the combined image
plt.savefig('kmeans_results/wordclouds/reddit_thread_schema_clusters.png', dpi=500)

# show the combined image
plt.show()


## REDDIT - CREATE SUBREDDIT AGGREGATION SCHEMA ##

# retain the following columns during aggregation
retain_columns = ['subreddit', 'lemmatized_content']

# perform aggregation
reddit_subreddit_aggregation = aggregate_into_schema(reddit_labeled, retain_columns, 'subreddit')

# text data
reddit_subreddit_text_data = reddit_subreddit_aggregation['lemmatized_content'].tolist()

# save paths - pca
reddit_subreddit_pca_paths = {'max': 'kmeans_results/pca_results/reddit_subreddit_max.png',
                              'tenth': 'kmeans_results/pca_results/reddit_subreddit_tenth.png',
                              'lda_3': 'kmeans_results/pca_results/reddit_subreddit_lda_3.png',
                              'lda_5': 'kmeans_results/pca_results/reddit_subreddit_lda_5.png'}

# save paths - kmeans
reddit_subreddit_kmeans_paths = {'max': 'kmeans_results/silhouette_results/reddit_subreddit_max.png',
                                 'tenth': 'kmeans_results/silhouette_results/reddit_subreddit_tenth.png',
                                 'lda_3': 'kmeans_results/silhouette_results/reddit_subreddit_lda_3.png',
                                 'lda_5': 'kmeans_results/silhouette_results/reddit_subredditlda_5.png',
                                 'average': 'kmeans_results/silhouette_results/reddit_subreddit_average.png'}


## REDDIT - SUBREDDIT - CLUSTERING ##
cv_dictionary_reddit_subreddit, kmeans_dictionary_reddit_subreddit = explore_clusters(text_data=reddit_subreddit_text_data,
                                                                                      vectorizer_type='tfid',
                                                                                      pca_save_paths=reddit_subreddit_pca_paths,
                                                                                      kmeans_save_paths=reddit_subreddit_kmeans_paths)

## REDDIT - SUBREDDIT - CLUSTER PICK ##
'''
- Max: 3 (or 4)
- Tenth: 4 (or 3)
- LDA 3: 2 (or 4)
- LDA 3: 4 (or 5)
'''
# visualize 3D
visualize_results_3d(kmeans_dictionary_reddit_subreddit['max'], 'clusters_3', 'Clusters', 'kmeans_results/reddit_subreddit_maximum_3_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_subreddit['tenth'], 'clusters_4', 'Clusters', 'kmeans_results/reddit_subreddit_tenth_4_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_subreddit['lda_3'], 'clusters_2', 'Clusters', 'kmeans_results/reddit_subreddit_lda_3_2_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_subreddit['lda_3'], 'clusters_4', 'Clusters', 'kmeans_results/reddit_subreddit_lda_3_4_clusters.html', opacity=0.7, arrow_size=10)
visualize_results_3d(kmeans_dictionary_reddit_subreddit['lda_5'], 'clusters_4', 'Clusters', 'kmeans_results/reddit_subreddit_lda_5_4_clusters.html', opacity=0.7, arrow_size=10)

'''
cluster choice:
    - Data is very spread out compared to the others, probably due to substantially less data
    - Iterative 3 Topic LDA with 2 Clusters
    - Highest coefficient value outside the results which have the major cluster as the mass in the origin and then a few points outside as their own clusters
    - Separates the origin mass well
'''

## REDDIT - SUBREDDIT - FINAL CLUSTER ILLUSTRATIONS ##

# illustrate cluster balance
illustrate_cluster_balance(kmeans_dictionary_reddit_subreddit['lda_3'], 'clusters_2', 'Iterative 3 Topic LDA', save_path='kmeans_results/reddit_subreddit_cluster_selection.png')

# concatenate the cluster choice into the main dataframe
reddit_subreddit_cluster_selection = kmeans_dictionary_reddit_subreddit['lda_3'][['clusters_2']]
reddit_subreddit_cluster_selection.rename(columns={'clusters_2': 'Cluster'}, inplace=True)
reddit_subreddit_clustered = pd.concat([reddit_labeled, reddit_subreddit_cluster_selection], axis=1)

# group and concatenate content by cluster
reddit_subreddit_cluster_grouped = reddit_subreddit_clustered.groupby('Cluster')['lemmatized_content'].apply(' '.join).reset_index()

# extract cluster content from grouping
cluster_content_0 = reddit_subreddit_cluster_grouped[reddit_subreddit_cluster_grouped['Cluster'] == 0]['lemmatized_content'].values[0]
cluster_content_1 = reddit_subreddit_cluster_grouped[reddit_subreddit_cluster_grouped['Cluster'] == 1]['lemmatized_content'].values[0]

# create word clouds for top 100 words
wc_0 = create_word_cloud(cluster_content_0, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')
wc_1 = create_word_cloud(cluster_content_1, width=1200, height=800, max_words=20, colormap='viridis', min_word_length=0, background_color='white', cloud_method='generate')

# plot the word clouds
fig, axs = plt.subplots(1, 2, figsize=(24, 8))

# wordcloud for cluster 0
axs[0].imshow(wc_0, interpolation='bilinear')
axs[0].axis('off')
axs[0].set_title('Reddit Subreddit Schema - Cluster 0')

# wordcloud for cluster 1
axs[1].imshow(wc_1, interpolation='bilinear')
axs[1].axis('off')
axs[1].set_title('Reddit Subreddit Schema - Cluster 1')

# Save the combined image
plt.savefig('kmeans_results/wordclouds/reddit_subreddit_schema_clusters.png', dpi=500)

# Show the combined image
plt.show()

## CREATE CORPUSES AND TRANSACTION DATA WITH INFORMATIVE DATA ##
'''
Informative Data:
    - NewsAPI: Iterative 3 Topic LDA
    - Reddit Base Schema: Iterative 5 Topic LDA
    - Reddit Author Aggregation Schema: Iterative 5 Topic LDA
    - Reddit Thread Aggregation Schema: Iterative 3 Topic LDA
    - Reddit Subreddit Aggregation Schema: Iterative 3 Topic LDA

Corpus Purpose: Clustering in R

Transaction Data Purpose: ARM in R

Further Processing:
    - NewsAPI Aggregation: aggregate by source

Corpus Filenames:
    - NewsAPI: iterative source naming map
    - NewsAPI Aggregation: source
    - Reddit Base Schema: iterative thread naming map with iterative author id naming map (due to unique usernames)
    - Reddit Author Schema: iterative author id naming map (due to unique usernames)
    - Reddit Thread Schema: iterative thread naming map
    - Reddit Subreddit Schema: subreddit

Transaction Data (Purpose: ARM in R)
'''

## NEWSAPI: REDUCE ARTICLES TO INFORMATIVE DATA ##
# wordset
wordset_newsapi = cv_dictionary_newsapi['lda_3'].columns.tolist()

# informative column on new copy
informative_newsapi = newsapi_labeled.copy()
informative_newsapi['informative'] = informative_newsapi['lemmatized_article'].apply(lambda x: retain_words(x, wordset_newsapi))

## NEWSAPI: CREATE NAMING MAP ##
# create source naming map
newsapi_source_map = create_naming_map(informative_newsapi, 'source', 'url')

# save source naming map - accounts for different ordering
newsapi_source_map.to_csv('naming_maps/newsapi_source_map.csv', index=False)

# merge in source naming map
informative_newsapi = pd.merge(informative_newsapi, newsapi_source_map[['source_id', 'url']], on='url')

## NEWSAPI: CREATE CORPUS ##
create_corpus(informative_newsapi, 'source_id', 'informative', 'corpus_newsapi')

## NEWSAPI: CREATE TRANSACTION DATA ##
vectorized_to_transaction(cv_dictionary_newsapi['lda_3'], '../arm/transaction_data/transaction_newsapi.csv')


## NEWSAPI SOURCE AGGREGATION: PERFORM AGGREGATION ##
informative_newsapi_aggregated = aggregate_into_schema(informative_newsapi, ['source', 'informative'], 'source')

## NEWSAPI SOURCE AGGREGATION: CREATE CORPUS ##
create_corpus(informative_newsapi_aggregated, 'source', 'informative', 'corpus_newsapi_aggregated')

## NEWSAPI SOURCE AGGREGATION: CREATE TRANSACTION DATA ##
column_to_transaction(informative_newsapi_aggregated, 'informative', '../arm/transaction_data/transaction_newsapi_aggregated.csv')


## REDDIT BASE SCHEMA: REDUCE ARTICLES TO INFORMATIVE DATA ##
# wordset
wordset_reddit_base = cv_dictionary_reddit_base['lda_5'].columns.tolist()

# informative column on new copy
informative_reddit_base = reddit_labeled.copy()
informative_reddit_base['informative'] = informative_reddit_base['lemmatized_content'].apply(lambda x: retain_words(x, wordset_reddit_base))

## REDDIT BASE SCHEMA: CREATE NAMING MAP ##
# create iterative threads map
reddit_threads = informative_reddit_base['url'].unique().tolist()
reddit_thread_ids = [f'thread_{num}' for num in range(len(reddit_threads))]
reddit_threads_map = pd.DataFrame({'url': reddit_threads, 'thread': reddit_thread_ids})

# save iterative theads map
reddit_threads_map.to_csv('naming_maps/reddit_threads_map.csv', index=False)

# merge in iterative threads map
informative_reddit_base = pd.merge(informative_reddit_base, reddit_threads_map, on='url')

# add primary key column (thread - author)
informative_reddit_base['primary_key'] = informative_reddit_base['thread'] + ' ' + informative_reddit_base['author']

## REDDIT BASE SCHEMA: CREATE CORPUS ##
create_corpus(informative_reddit_base, 'primary_key', 'informative', 'corpus_reddit_base')

## REDDIT BASE SCHEMA: CREATE TRANSACTION DATA ##
vectorized_to_transaction(cv_dictionary_reddit_base['lda_5'], '../arm/transaction_data/transaction_reddit_base.csv')


## REDDIT AUTHOR SCHEMA: REDUCE ARTICLES TO INFORMATIVE DATA ##
# wordset
wordset_reddit_author = cv_dictionary_reddit_author['lda_5'].columns.tolist()

# informative column on new copy
informative_reddit_author = reddit_author_aggregation.copy()
informative_reddit_author['informative'] = informative_reddit_author['lemmatized_content'].apply(lambda x: retain_words(x, wordset_reddit_author))

## REDDIT AUTHOR SCHEMA: CREATE CORPUS ##
create_corpus(informative_reddit_author, 'author', 'informative', 'corpus_reddit_author')

## REDDIT AUTHOR SCHEMA: CREATE TRANSACTION DATA ##
vectorized_to_transaction(cv_dictionary_reddit_author['lda_5'], '../arm/transaction_data/transaction_reddit_author.csv')


## REDDIT THREAD SCHEMA: REDUCE ARTICLES TO INFORMATIVE DATA ##
# wordset
wordset_reddit_thread = cv_dictionary_reddit_thread['lda_3'].columns.tolist()

# informative column on new copy
informative_reddit_thread = reddit_thread_aggregation.copy()
informative_reddit_thread['informative'] = informative_reddit_thread['lemmatized_content'].apply(lambda x: retain_words(x, wordset_reddit_thread))

# merge in thread naming map
informative_reddit_thread = pd.merge(informative_reddit_thread, reddit_threads_map, on='url')

## REDDIT THREAD SCHEMA: CREATE CORPUS ##
create_corpus(informative_reddit_thread, 'thread', 'informative', 'corpus_reddit_thread')

## REDDIT THREAD SCHEMA: CREATE TRANSACTION DATA ##
vectorized_to_transaction(cv_dictionary_reddit_thread['lda_3'], '../arm/transaction_data/transaction_reddit_thread.csv')


## REDDIT SUBREDDIT SCHEMA: REDUCE ARTICLES TO INFORMATIVE DATA ##
# wordset
wordset_reddit_subreddit = cv_dictionary_reddit_subreddit['lda_3'].columns.tolist()

# informative column on new copy
informative_reddit_subreddit = reddit_subreddit_aggregation.copy()
informative_reddit_subreddit['informative'] = informative_reddit_subreddit['lemmatized_content'].apply(lambda x: retain_words(x, wordset_reddit_subreddit))

## REDDIT SUBREDDIT SCHEMA: CREATE CORPUS ##
create_corpus(informative_reddit_subreddit, 'subreddit', 'informative', 'corpus_reddit_subreddit')

## REDDIT SUBREDDIT SCHEMA: CREATE TRANSACTION DATA ##
vectorized_to_transaction(cv_dictionary_reddit_subreddit['lda_3'], '../arm/transaction_data/transaction_reddit_subreddit.csv')
