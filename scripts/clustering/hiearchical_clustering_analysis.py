'''
Hiearchcical Clustering Analysis
'''

# import specific functions
from exploratory_functions import *
from vectorizing_functions import *

'''
INITIAL ANALYSIS
'''

# import hierarchical clustering results
hierarchical_newsapi = pd.read_csv('hierarchical_results/hierarchical_newsapi.csv')

# examine how the articles are sorted by source
hierarchical_newsapi['source_id'] = hierarchical_newsapi['File'].apply(lambda x: x.replace('.txt', ''))
hierarchical_newsapi['source'] = hierarchical_newsapi['source_id'].apply(lambda x: x.split('_')[0])

# bring in political leaning maps
newsapi_labeled = pd.read_csv('../send_files/cleaning/newsapi_data_cleaned/newsapi_labeled_2_14_25.csv')
newsapi_labeled.dropna(subset='Bias Specific', inplace=True)
newsapi_labeled.rename(columns={'Bias Specific': 'political_bias'}, inplace=True)
newsapi_labeled['simple_bias'] = newsapi_labeled['political_bias'].replace({'Lean Left': 'Left', 'Lean Right': 'Right'})
political_leaning_map = newsapi_labeled[['source', 'political_bias', 'simple_bias']].drop_duplicates(subset='source', ignore_index=True)

# same labels together while different labels clustered apart?
hierarchical_newsapi = pd.merge(hierarchical_newsapi, political_leaning_map, on='source')
majority_bias_df = calculate_multiple_majorities(hierarchical_newsapi, 'simple_bias')

# analyze similar within labels and disimilar between labels
cluster_columns = [col for col in majority_bias_df.columns if col.startswith('cluster_')]
proportion_columns = [col for col in majority_bias_df.columns if col.startswith('proportion_')]
majority_cluster_melted = majority_bias_df[['simple_bias'] + cluster_columns].melt('simple_bias')
majority_proportion_melted = majority_bias_df[['simple_bias'] + proportion_columns].melt('simple_bias')
majority_melted = pd.concat([majority_cluster_melted.rename(columns={'variable': 'cluster', 'value': 'cluster_value'}), majority_proportion_melted.rename(columns={'variable': 'proportion', 'value': 'proportion_value'})[['proportion', 'proportion_value']]], axis=1)
majority_melted['cluster_cut'] = majority_melted['cluster'].apply(lambda x: int(x.split('_')[1]))

# save majority_melted
majority_melted.to_csv('hierarchical_results/majority_newsapi.csv', index=False)

# plotting
sns.lineplot(majority_melted, x='cluster_cut', y='cluster_value', hue='simple_bias')
plt.yticks([0, 1, 2, 3, 4])
plt.xticks([2, 3, 4, 5, 6])
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Political Bias')
plt.xlabel('Hierarchical Level')
plt.ylabel('Hierarchical Cluster')
plt.title('NewsAPI Cluster Majorities')
plt.savefig('hierarchical_results/majority_newsapi_divergence.png', dpi=300, bbox_inches='tight')
plt.show()

# cluster 6 separates well
'''
- center: cluster 1 @ 52.2%
- left: cluster 2 @ 37.9%
- right: cluster 4 @ 28.6%
'''

# where elese can we partition the labels?
simple_proportion_6 = proportion_clusters_by_label(hierarchical_newsapi, 'simple_bias', 'cluster_6')

# visualize - pair the above plots together
sns.barplot(simple_proportion_6, x='cluster_6', y='proportion', hue='simple_bias')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Political Bias')
plt.xlabel('Cluster')
plt.ylabel('Proportion of Bias Label')
plt.title('NewsAPI Bias Proportions in Clusters')
plt.savefig('hierarchical_results/newsapi_cluster_proportions.png', dpi=300, bbox_inches='tight')
plt.show()

# further inspect
'''
- center: cluster 1 @ 52.2%, cluster 5 @ 12.3%, cluster 6 @ 12.3%
- left: cluster 2 @ 37.9%, cluster 3 @ 18.5%, cluster 5 @ 16.1%
- right: cluster 4 @ 28.6%, cluster 3 @ 22.1%, cluster 1 @ 18.2%

breaking this down:
    - center: 1, 5, 6 -> 1, 6
    - left: 2, 3, 5 -> 2, 5
    - right: 4, 3, 1 -> 4, 3
'''

# this can be further progressed by assigning a second cluster to each political bias label
# center: 1 & 6
# left: 2 & 5
# right: 4 & 3
# subset
simple_proportion_center = simple_proportion_6[(simple_proportion_6['cluster_6'].isin([1, 6])) &
                                               (simple_proportion_6['simple_bias'] == 'Center')]
simple_proportion_left = simple_proportion_6[(simple_proportion_6['cluster_6'].isin([2, 5])) &
                                             (simple_proportion_6['simple_bias'] == 'Left')]
simple_proportion_right = simple_proportion_6[(simple_proportion_6['cluster_6'].isin([3, 4])) &
                                              (simple_proportion_6['simple_bias'] == 'Right')]
# assign
simple_proportion_assigned = pd.concat([simple_proportion_center, simple_proportion_left], ignore_index=True)
simple_proportion_assigned = pd.concat([simple_proportion_assigned, simple_proportion_right], ignore_index=True)

# save
simple_proportion_assigned.to_csv('hierarchical_results/majority_newsapi_assigned.csv', index=False)

# illustrate - set up
color_palette = sns.color_palette('Paired')
cluster_assignments = {'Center': [1, 6], 'Left': [2, 5], 'Right': [4, 3]}
color_assignments = {'Center': [1, 0], 'Left': [3, 2], 'Right': [5, 4]}

# illustrate - paired colors
bar_colors = {'Bias': [], 'Cluster': [], 'Color': []}
for bias in cluster_assignments:
    for pair_num, cluster in enumerate(cluster_assignments[bias]):
        bar_colors['Bias'].append(bias)
        bar_colors['Cluster'].append(cluster)
        bar_colors['Color'].append(color_palette[color_assignments[bias][pair_num]])

# illustrate - sort and create ordered colored list
bar_colors_df = pd.DataFrame(bar_colors)
bar_colors_sorted = bar_colors_df.sort_values('Cluster')['Color'].tolist()

# illustrate - create plot
assigned_pivot.plot(kind='bar', stacked=True, color=bar_colors_sorted)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Cluster Pairs')
plt.xlabel('Politcal Bias')
plt.xticks(rotation=0)
plt.ylabel('Proportion of Bias Label')
plt.title('NewsAPI Paired Bias Proportions in Clusters')
plt.savefig('hierarchical_results/newsapi_cluster_proportions_paired.png', dpi=300, bbox_inches='tight')
plt.show()


'''
SUBSET ARTICLES
'''

# which articles can we extract from these clusters
subset_left = hierarchical_newsapi[(hierarchical_newsapi['cluster_6'].isin([2, 5])) &
                                   (hierarchical_newsapi['simple_bias'] == 'Left')]
subset_center = hierarchical_newsapi[(hierarchical_newsapi['cluster_6'].isin([1, 6])) &
                                     (hierarchical_newsapi['simple_bias'] == 'Center')]
subset_right = hierarchical_newsapi[(hierarchical_newsapi['cluster_6'].isin([4, 3])) &
                                    (hierarchical_newsapi['simple_bias'] == 'Right')]

combined_subsets = pd.concat([subset_left, subset_center], ignore_index=True)
combined_subsets = pd.concat([combined_subsets, subset_right], ignore_index=True)

combined_proportion_6 = proportion_clusters_by_label(combined_subsets, 'simple_bias', 'cluster_6')

## NOTE: Check Length of Articles for Reddit Comparison ##
# get source map
source_map = pd.read_csv('naming_maps/newsapi_source_map.csv')
newsapi_labeled = pd.merge(newsapi_labeled, source_map[['url', 'source_id']], on='url')
combined_sources = combined_subsets['source_id'].unique().tolist()

# similar within and dissimilar between
newsapi_subset = newsapi_labeled[newsapi_labeled['source_id'].isin(combined_sources)]
newsapi_subset.reset_index(drop=True, inplace=True)


'''
RERUN FOR LDA3 WORDSET
'''
# apply function - further specific cleaning
newsapi_labeled['cleaned_article'] = newsapi_labeled['article'].apply(specific_cleaning)

# apply function - lemmatize
newsapi_labeled['lemmatized_article'] = newsapi_labeled['cleaned_article'].apply(lemmatize_article)

# remove additional words
newsapi_labeled['lemmatized_article'] = newsapi_labeled['lemmatized_article'].apply(lambda x: remove_additional_words(x, ['wa', 'ha', 'tt', 've', 'wt', 'tn']))

# subset text data
text_data = newsapi_labeled['lemmatized_article'].tolist()

params_max = {'stop_words': 'english'}
cv_max = vectorize_to_df(text_data, input_type='content', vectorizer_type='count', params=params_max)

# 10% maximum features
params_tenth = {'stop_words': 'english',
                'max_features': int(cv_max.shape[1] / 10)}
cv_tenth = vectorize_to_df(text_data, input_type='content', vectorizer_type='count', params=params_tenth)

# lda 3 topics
lda_model_3, lda_object_3 = run_lda(df=cv_tenth, num_topics=3, iterations=100, learning='online')
top_words_list_3, top_words_fd_3 = return_top_lda(lda_model_3, num_words=50)
cv_lda_3 = cv_tenth[top_words_list_3]

# wordset
wordset = cv_lda_3.columns.tolist()

# save cv_lda_3
cv_lda_3.to_csv('newsapi_optimal_vectorized.csv', index=False)

'''
APPLY WORDSET TO NEWSAPI SUBSET
'''

# apply function - further specific cleaning
newsapi_subset['cleaned_article'] = newsapi_subset['article'].apply(specific_cleaning)

# apply function - lemmatize
newsapi_subset['lemmatized_article'] = newsapi_subset['cleaned_article'].apply(lemmatize_article)

# remove additional words
newsapi_subset['lemmatized_article'] = newsapi_subset['lemmatized_article'].apply(lambda x: remove_additional_words(x, ['wa', 'ha', 'tt', 've', 'wt', 'tn']))

# subset on the wordset
newsapi_subset['informative'] = newsapi_subset['lemmatized_article'].apply(lambda x: retain_words(x, wordset))

# lengths
newsapi_subset['informative_lengths'] = newsapi_subset['informative'].apply(lambda x: len(x.split()))

# illustrate lengths
sns.displot(newsapi_subset, x='informative_lengths')

# get minimum length
minimum_informative = newsapi_subset['informative_lengths'].min()

'''
FIND REDDIT SCHEMA WHICH CLOSELY MATCHES THE DISTRIBUTION
'''

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

## REDDIT BASE SCHEMA ##
# create copy
reddit_base_schema = reddit_labeled.copy()

# subset on the wordset
reddit_base_schema['informative'] = reddit_base_schema['lemmatized_content'].apply(lambda x: retain_words(x, wordset))

# lengths
reddit_base_schema['informative_lengths'] = reddit_base_schema['informative'].apply(lambda x: len(x.split()))

# remove rows with informative_lengths less than newsapi's
reddit_base_schema = reddit_base_schema[reddit_base_schema['informative_lengths'] >= minimum_informative]
reddit_base_schema.reset_index(drop=True, inplace=True)

# ilustrate
sns.displot(reddit_base_schema, x='informative_lengths')

## REDDIT AUTHOR SCHEMA ##
# aggregate into schema
retain_columns = ['author', 'informative']
reddit_author_schema = aggregate_into_schema(reddit_base_schema, retain_columns, 'author')

# lengths
reddit_author_schema['informative_lengths'] = reddit_author_schema['informative'].apply(lambda x: len(x.split()))

# remove rows with informative_lengths less than newsapi's
reddit_author_schema = reddit_author_schema[reddit_author_schema['informative_lengths'] >= minimum_informative]
reddit_author_schema.reset_index(drop=True, inplace=True)

# illustrate
sns.displot(reddit_author_schema, x='informative_lengths')

## REDDIT THREAD SCHEMA ##
# aggregate into schema
retain_columns = ['url', 'informative']
reddit_thread_schema = aggregate_into_schema(reddit_base_schema, retain_columns, 'url')

# lengths
reddit_thread_schema['informative_lengths'] = reddit_thread_schema['informative'].apply(lambda x: len(x.split()))

# remove rows with informative_lengths less than newsapi's
reddit_thread_schema = reddit_thread_schema[reddit_thread_schema['informative_lengths'] >= minimum_informative]
reddit_thread_schema.reset_index(drop=True, inplace=True)

# illustrate
sns.displot(reddit_thread_schema, x='informative_lengths')

## REDDIT SUBREDDIT SCHEMA ##
# aggregate into schema
retain_columns = ['subreddit', 'informative']
reddit_subreddit_schema = aggregate_into_schema(reddit_base_schema, retain_columns, 'subreddit')

# lengths
reddit_subreddit_schema['informative_lengths'] = reddit_subreddit_schema['informative'].apply(lambda x: len(x.split()))

# remove rows with informative_lengths less than newsapi's
reddit_subreddit_schema = reddit_subreddit_schema[reddit_subreddit_schema['informative_lengths'] >= minimum_informative]
reddit_subreddit_schema.reset_index(drop=True, inplace=True)

# illustrate
sns.displot(reddit_subreddit_schema, x='informative_lengths')

## ILLUSTRATE INFORMATIVE DISTRIBUTIONS ##
# create dataframe to allow discovery - NewsAPI
informative_newsapi = newsapi_subset[['informative_lengths']]
informative_newsapi['origin'] = 'NewsAPI'

# create dataframe to allow discovery - Reddit Base Schema
informative_reddit_base = reddit_base_schema[['informative_lengths']]
informative_reddit_base['origin'] = 'Reddit Base Schema'

# create dataframe to allow discovery - Reddit Author Schema
informative_reddit_author = reddit_author_schema[['informative_lengths']]
informative_reddit_author['origin'] = 'Reddit Author Schema'

# create dataframe to allow discovery - Reddit Thread Schema
informative_reddit_thread = reddit_thread_schema[['informative_lengths']]
informative_reddit_thread['origin'] = 'Reddit Thread Schema'

# create dataframe to allow discovery - Reddit Subreddit Schema
informative_reddit_subreddit = reddit_subreddit_schema[['informative_lengths']]
informative_reddit_subreddit['origin'] = 'Reddit Subreddit Schema'

# create dataframe to allow discovery - concatenate
informative_concatenated = pd.concat([informative_newsapi, informative_reddit_base], ignore_index=True)
informative_concatenated = pd.concat([informative_concatenated, informative_reddit_author], ignore_index=True)
informative_concatenated = pd.concat([informative_concatenated, informative_reddit_thread], ignore_index=True)
informative_concatenated = pd.concat([informative_concatenated, informative_reddit_subreddit], ignore_index=True)

# reddit origins
reddit_origins = ['Reddit Base Schema', 'Reddit Author Schema', 'Reddit Thread Schema', 'Reddit Subreddit Schema']

# illustrate lengths
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# flatten the axs array for easier indexing
axes = axs.flatten()

# iterate through the reddit origins to compare
for plot_num, origin in enumerate(reddit_origins):
    informative_subset = informative_concatenated[informative_concatenated['origin'].isin(['NewsAPI', origin])]
    sns.boxplot(informative_subset, x='informative_lengths', hue='origin', ax=axes[plot_num])
    axes[plot_num].legend(title='Origin')
    axes[plot_num].set_xlabel('')
    axes[plot_num].set_ylabel('')

# label and title options
# fig.text(0.5, 0, '\nContent Length with Informative Wordset', ha='center', va='center', fontsize=20)
# fig.text(0, 0.5, 'Origin', ha='center', va='center', rotation='vertical', fontsize=20)
fig.text(0.5, 1, 'Informative Content Lengths Comparison\n', ha='center', va='center', fontsize=20)

# show image
plt.tight_layout()
plt.savefig('hierarchical_results/informative_content_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

'''
Now we want to balance the sets with random sampling for the varying aggregation schemas:
    - Downsample Reddit Base
    - Downsample Reddit Author
    - Aggregate NewsAPI Subset by Source
'''
# newsapi subset size
sample_size = newsapi_subset.shape[0]

# balance reddit base schema
reddit_base_schema_subset = reddit_base_schema.sample(n=sample_size, random_state=42, ignore_index=True)

# balance reddit author schema
reddit_author_schema_subset = reddit_author_schema.sample(n=sample_size, random_state=42, ignore_index=True)

# aggregate newsapi subset
retain_columns = ['source', 'informative']
newsapi_subset_source = aggregate_into_schema(newsapi_subset, retain_columns, 'source')

## RERUN INFORMATIVE COMPARISON ##
# calculate new informative lengths for source aggregation
newsapi_subset_source['informative_lengths'] = newsapi_subset_source['informative'].apply(lambda x: len(x.split()))

# create dataframe to allow discovery - NewsAPI
informative_newsapi_source = newsapi_subset_source[['informative_lengths']]
informative_newsapi_source['origin'] = 'NewsAPI Source Schema'

# create dataframe to allow discovery - Reddit Base Schema
informative_reddit_base_subset = reddit_base_schema_subset[['informative_lengths']]
informative_reddit_base_subset['origin'] = 'Reddit Base Schema Downsample'

# create dataframe to allow discovery - Reddit Author Schema
informative_reddit_author_subset = reddit_author_schema_subset[['informative_lengths']]
informative_reddit_author_subset['origin'] = 'Reddit Author Schema Downsample'

# create dataframe to allow discovery - Reddit Thread Schema (KEEP FROM BEFORE)
informative_reddit_thread = reddit_thread_schema[['informative_lengths']]
informative_reddit_thread['origin'] = 'Reddit Thread Schema'

# create dataframe to allow discovery - Reddit Subreddit Schema (KEEP FROM BEFORE)
informative_reddit_subreddit = reddit_subreddit_schema[['informative_lengths']]
informative_reddit_subreddit['origin'] = 'Reddit Subreddit Schema'

# create dataframe to allow discovery - concatenate
informative_concatenated_balanced = pd.concat([informative_newsapi, informative_newsapi_source], ignore_index=True)
informative_concatenated_balanced = pd.concat([informative_concatenated_balanced, informative_reddit_base_subset], ignore_index=True)
informative_concatenated_balanced = pd.concat([informative_concatenated_balanced, informative_reddit_author_subset], ignore_index=True)
informative_concatenated_balanced = pd.concat([informative_concatenated_balanced, informative_reddit_thread], ignore_index=True)
informative_concatenated_balanced = pd.concat([informative_concatenated_balanced, informative_reddit_subreddit], ignore_index=True)

# reddit origins
reddit_origins_balanced = ['Reddit Base Schema Downsample', 'Reddit Author Schema Downsample', 'Reddit Thread Schema', 'Reddit Subreddit Schema']

# illustrate lengths
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# flatten the axs array for easier indexing
axes = axs.flatten()

# iterate through the reddit origins to compare
for plot_num, origin in enumerate(reddit_origins_balanced):
    informative_subset = informative_concatenated_balanced[informative_concatenated_balanced['origin'].isin(['NewsAPI', 'NewsAPI Source Schema', origin])]
    sns.boxplot(informative_subset, x='informative_lengths', hue='origin', ax=axes[plot_num])
    axes[plot_num].legend(title='Origin')
    axes[plot_num].set_xlabel('')
    axes[plot_num].set_ylabel('')

fig.text(0.5, 1, 'Balanced Informative Content Lengths Comparison\n', ha='center', va='center', fontsize=20)

# show image
plt.tight_layout()
plt.savefig('hierarchical_results/informative_content_comparison_balanced.png', dpi=300, bbox_inches='tight')
plt.show()

'''
This balances the samples used, but the distributions still vary wildly.
'''

## CONCATENATED DATAFRAMES TO PERFORM HIERARCHICAL CLUSTERING WITH ##

'''
newsapi subset with reddit base schema downsample
'''

# newsapi_subset with primary key
newsapi_subset_primary = newsapi_subset[['source_id', 'informative']]

# create new key to specify newsapi
newsapi_subset_primary['new_key'] = 'NewsAPI ' + newsapi_subset_primary['source_id']

# reddit base downsample with primary key
# reddit thread map
thread_map = pd.read_csv('naming_maps/reddit_threads_map.csv')

# merge in thread map
reddit_base_schema_primary = pd.merge(reddit_base_schema_subset, thread_map, on='url')

# create primary key
reddit_base_schema_primary['primary_key'] = reddit_base_schema_primary['thread'] + ' ' + reddit_base_schema_primary['author']

# create new key to specifiy reddit
reddit_base_schema_primary['new_key'] = 'Reddit ' + reddit_base_schema_primary['primary_key']

# first concatenated dataframe
key_columns = ['new_key', 'informative']
newsapi_reddit_base = pd.concat([newsapi_subset_primary[key_columns], reddit_base_schema_primary[key_columns]], ignore_index=True)

'''
newsapi subset with reddit author schema downsample
'''

# create new key to specify reddit
reddit_author_schema_primary = reddit_author_schema_subset[['author', 'informative']]
reddit_author_schema_primary['new_key'] = 'Reddit ' + reddit_author_schema_primary['author']

# second concatenated dataframe
key_columns = ['new_key', 'informative']
newsapi_reddit_author = pd.concat([newsapi_subset_primary[key_columns], reddit_author_schema_primary[key_columns]], ignore_index=True)

'''
newsapi subset source schema with reddit thread schema
'''

# create new key to specifcy newsapi
newsapi_subset_source_primary = newsapi_subset_source.copy()
newsapi_subset_source_primary['new_key'] = 'NewsAPI ' + newsapi_subset_source_primary['source']

# merge in thread map
reddit_thread_schema_primary = pd.merge(reddit_thread_schema, thread_map, on='url')

# create new key to specify reddit
reddit_thread_schema_primary['new_key'] = 'Reddit ' + reddit_thread_schema_primary['thread']

# third concatenated dataframe
key_columns = ['new_key', 'informative']
newsapi_reddit_thread = pd.concat([newsapi_subset_source_primary[key_columns], reddit_thread_schema_primary[key_columns]], ignore_index=True)

'''
newsapi subset source schema with reddit subreddit schema
'''

# create new key to specify reddit
reddit_subreddit_schema_primary = reddit_subreddit_schema.copy()
reddit_subreddit_schema_primary['new_key'] = 'Reddit ' + reddit_subreddit_schema_primary['subreddit']

# fourth concatenated dataframe
key_columns = ['new_key', 'informative']
newsapi_reddit_subreddit = pd.concat([newsapi_subset_source_primary[key_columns], reddit_subreddit_schema_primary[key_columns]], ignore_index=True)

'''
CORPUSES SECTION
'''

## CORPUSES FOR HIERARCHICAL CLUSTERING IN R ##

# corpus for newsapi subset with reddit base schema downsample
create_corpus(newsapi_reddit_base, 'new_key', 'informative', 'corpus_newsapi_reddit_base')

# corpus for newsapi subset with reddit author schema downsample
create_corpus(newsapi_reddit_author, 'new_key', 'informative', 'corpus_newsapi_reddit_author')

# corpus for newsapi subset source schema with reddit thread schema downsample
create_corpus(newsapi_reddit_thread, 'new_key', 'informative', 'corpus_newsapi_reddit_thread')

# corpus for newsapi subset source schema with reddit subreddit schema downsample
create_corpus(newsapi_reddit_subreddit, 'new_key', 'informative', 'corpus_newsapi_reddit_subreddit')


'''
SECONDARY ANALYSIS - NEWSAPI + REDDIT HIEARCHICAL RESULTS
'''

# function to extract the keys
def extract_keys(hierarchical_df):
    extracted_df = hierarchical_df.copy()
    extracted_df['origin'] = extracted_df['File'].apply(lambda x: x.split()[0])
    extracted_df['original_key'] = extracted_df.apply(lambda row: row['File'].replace(f"{row['origin']} ", ''), axis=1)
    extracted_df['original_key'] = extracted_df['original_key'].apply(lambda x: x.replace('.txt', ''))
    
    return extracted_df

# import hierarchical clustering results
cluster_newsapi_reddit_author = pd.read_csv('hierarchical_results/hierarchical_newsapi_reddit_author.csv')

# apply extract keys
cluster_newsapi_reddit_author = extract_keys(cluster_newsapi_reddit_author)
cluster_newsapi_reddit_author['source'] = cluster_newsapi_reddit_author.apply(lambda row: row['original_key'].split('_')[0] if row['origin']=='NewsAPI' else row['original_key'], axis=1)

# proportions
newsapi_reddit_author_majorities = calculate_multiple_majorities(cluster_newsapi_reddit_author, 'origin')
newsapi_reddit_author_proportion_2 = proportion_clusters_by_label(cluster_newsapi_reddit_author, 'origin', 'cluster_2')
newsapi_reddit_author_proportion_3 = proportion_clusters_by_label(cluster_newsapi_reddit_author, 'origin', 'cluster_3')

# illustrate
sns.barplot(newsapi_reddit_author_proportion_3, x='cluster_3', y='proportion', hue='origin')
plt.xlabel('Cluster')
plt.ylabel('Proportion')
plt.title('NewsAPI vs. Reddit Author Schema')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Origin')
plt.tight_layout()
plt.savefig('hierarchical_results/newsapi_author_cluster_proportions.png', dpi=300, bbox_inches='tight')
plt.show()

'''
The expectation was to see the sources which had previously clustered together under politcal bias to stick together and then 
potentially give an indication of where a Reddit author could've fallen on the political spectrum.

in reality, a bifurcation of news articles and Reddit posts were formed.
'''

# import hierarchical clustering results
cluster_newsapi_reddit_thread = pd.read_csv('hierarchical_results/hierarchical_newsapi_reddit_thread.csv')

# apply extract keys
cluster_newsapi_reddit_thread = extract_keys(cluster_newsapi_reddit_thread)
cluster_newsapi_reddit_thread['source'] = cluster_newsapi_reddit_thread.apply(lambda row: row['original_key'].split('_')[0] if row['origin']=='NewsAPI' else row['original_key'], axis=1)

# proportions
newsapi_reddit_thread_majorities = calculate_multiple_majorities(cluster_newsapi_reddit_thread, 'origin')
newsapi_reddit_thread_proportion_2 = proportion_clusters_by_label(cluster_newsapi_reddit_thread, 'origin', 'cluster_2')
newsapi_reddit_thread_proportion_3 = proportion_clusters_by_label(cluster_newsapi_reddit_thread, 'origin', 'cluster_3')
newsapi_reddit_thread_proportion_4 = proportion_clusters_by_label(cluster_newsapi_reddit_thread, 'origin', 'cluster_4')

# illustrate
sns.barplot(newsapi_reddit_thread_proportion_2, x='cluster_2', y='proportion', hue='origin')
plt.xlabel('Cluster')
plt.ylabel('Proportion')
plt.title('NewsAPI Source Schema vs. Reddit Thread Schema')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Origin')
plt.tight_layout()
plt.savefig('hierarchical_results/newsapi_thread_cluster_proportions.png', dpi=300, bbox_inches='tight')
plt.show()

# import hierarchical clustering results
cluster_newsapi_reddit_subreddit = pd.read_csv('hierarchical_results/hierarchical_newsapi_reddit_subreddit.csv')

# apply extract keys
cluster_newsapi_reddit_subreddit = extract_keys(cluster_newsapi_reddit_subreddit)
cluster_newsapi_reddit_subreddit['source'] = cluster_newsapi_reddit_subreddit.apply(lambda row: row['original_key'].split('_')[0] if row['origin']=='NewsAPI' else row['original_key'], axis=1)

# proportions
newsapi_reddit_subreddit_majorities = calculate_multiple_majorities(cluster_newsapi_reddit_subreddit, 'origin')
newsapi_reddit_subreddit_proportion_2 = proportion_clusters_by_label(cluster_newsapi_reddit_subreddit, 'origin', 'cluster_2')
newsapi_reddit_subreddit_proportion_3 = proportion_clusters_by_label(cluster_newsapi_reddit_subreddit, 'origin', 'cluster_3')
newsapi_reddit_subreddit_proportion_4 = proportion_clusters_by_label(cluster_newsapi_reddit_subreddit, 'origin', 'cluster_4')

# illustrate
sns.barplot(newsapi_reddit_subreddit_proportion_2, x='cluster_2', y='proportion', hue='origin')
plt.xlabel('Cluster')
plt.ylabel('Proportion')
plt.title('NewsAPI Source Schema vs. Reddit Subreddit Schema')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Origin')
plt.tight_layout()
plt.savefig('hierarchical_results/newsapi_subreddit_cluster_proportions.png', dpi=300, bbox_inches='tight')
plt.show()
