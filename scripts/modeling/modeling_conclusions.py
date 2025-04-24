'''
Modeling Conclusions
'''

# import modeling functions
from modeling_functions import *

# import libraries not within modeling_functions
import pickle


'''
PART 1: IMPORT DATA
'''

# import data - newsapi countvectorized dataframes
newsapi_five = pd.read_csv('data/newsapi_five_cv.csv', index_col='Unnamed: 0')
newsapi_three = pd.read_csv('data/newsapi_three_cv.csv', index_col='Unnamed: 0')
newsapi_strict_three = pd.read_csv('data/newsapi_strict_three_cv.csv', index_col='Unnamed: 0')
newsapi_two = pd.read_csv('data/newsapi_two_cv.csv', index_col='Unnamed: 0')
newsapi_strict_two = pd.read_csv('data/newsapi_strict_two_cv.csv', index_col='Unnamed: 0')

# import data - reddit author aggregation schema and countvectorized dataframe
reddit_author_reduced = pd.read_csv('data/reddit_author_reduced.csv')
reddit_cv = pd.read_csv('data/reddit_author_cv.csv')


'''
PART 2: IMPORT RESULTS
'''

# nb feature importance through permutation importance
nb_features_three = pd.read_csv('nb/feature_importance_three.csv')
nb_features_two = pd.read_csv('nb/feature_importance_two.csv')

# dt feature importance through permutation importance
dt_features_two = pd.read_csv('dt/feature_importance_two.csv')

# svm feature importance through permutation importance
svm_features_three = pd.read_csv('svm/feature_importance_three.csv')
svm_features_two = pd.read_csv('svm/feature_importance_two.csv')

# nb models - three labels
with open('nb/nb_three.pkl', 'rb') as file:
    nb_model_three = pickle.load(file)

# nb models - two labels
with open('nb/nb_two.pkl', 'rb') as file:
    nb_model_two = pickle.load(file)

# dt models - two labels
with open('dt/dt_two.pkl', 'rb') as file:
    dt_model_two = pickle.load(file)
    
# svm models - three labels
with open('svm/svm_three.pkl', 'rb') as file:
    svm_model_three = pickle.load(file)

# svm models - two labels
with open('svm/svm_two.pkl', 'rb') as file:
    svm_model_two = pickle.load(file)

# import reddit results
nb_reddit_results = pd.read_csv('nb/nb_reddit_results.csv')
dt_reddit_results = pd.read_csv('dt/dt_reddit_results.csv')
svm_reddit_results = pd.read_csv('svm/svm_reddit_results.csv')


'''
PART 3: ANALYZE RESULTS - REDDIT USERS
'''

# edit reddit results for merging - drop individual conclusions
nb_reddit_results.drop(columns=['Threshold', 'Conclusion'], inplace=True)
dt_reddit_results.drop(columns=['Threshold', 'Conclusion'], inplace=True)
svm_reddit_results.drop(columns=['Threshold', 'Conclusion'], inplace=True)

# edit reddit results for merging - rename columns - nb
nb_reddit_results.rename(columns={'Predicted Bias Three': 'NB Bias Three',
                                  'Predicted Bias Two': 'NB Bias Two',
                                  'Threshold Three': 'NB Threshold Three',
                                  'Threshold Two': 'NB Threshold Two'}, inplace=True)
# edit reddit results for merging - rename columns - dt
dt_reddit_results.rename(columns={'Predicted Bias Three': 'DT Bias Three',
                                  'Predicted Bias Two': 'DT Bias Two',
                                  'Threshold Three': 'DT Threshold Three',
                                  'Threshold Two': 'DT Threshold Two'}, inplace=True)
# edit reddit results for merging - rename columns - svm
svm_reddit_results.rename(columns={'Predicted Bias Three': 'SVM Bias Three',
                                   'Predicted Bias Two': 'SVM Bias Two',
                                   'Threshold Three': 'SVM Threshold Three',
                                   'Threshold Two': 'SVM Threshold Two'}, inplace=True)

# combine reddit results
reddit_results = pd.merge(nb_reddit_results, dt_reddit_results, on='Author')
reddit_results = pd.merge(reddit_results, svm_reddit_results, on='Author')

'''
Columns:
    - Author
    - Model (NB, DT, SVM)
    - Tier (Two, Three)
    - Bias (Left, Center, Right)
    - Threshold
Strategy:
    - Melt on Biases
    - Melt on Thresholds
    - Merge and Reorder the Melts
'''

# first melt - biases
reddit_melted_1 = reddit_results[['Author'] + [col for col in reddit_results.columns if 'Bias' in col]].melt(id_vars=['Author'])
reddit_melted_1['Model'] = reddit_melted_1['variable'].apply(lambda x: x.split()[0])
reddit_melted_1['Tier'] = reddit_melted_1['variable'].apply(lambda x: x.split()[2])
reddit_melted_1.drop(columns=['variable'], inplace=True)
reddit_melted_1.rename(columns={'value': 'Bias'}, inplace=True)

# second melt - thresholds
reddit_melted_2 = reddit_results[['Author'] + [col for col in reddit_results.columns if 'Threshold' in col]].melt(id_vars=['Author'])
reddit_melted_2['Model'] = reddit_melted_2['variable'].apply(lambda x: x.split()[0])
reddit_melted_2['Tier'] = reddit_melted_2['variable'].apply(lambda x: x.split()[2])
reddit_melted_2.drop(columns=['variable'], inplace=True)
reddit_melted_2.rename(columns={'value': 'Threshold'}, inplace=True)

# merge melted
reddit_melted = pd.merge(reddit_melted_1, reddit_melted_2, on=['Author', 'Model', 'Tier'])

# reorder the melt
reddit_melted = reddit_melted[['Author', 'Model', 'Tier', 'Bias', 'Threshold']]

# analysis - two tier melt
reddit_two = reddit_melted[reddit_melted['Tier']=='Two'].groupby(['Author', 'Bias']).agg(
    two_bias_count = ('Bias', 'count'),
    two_bias_prob = ('Threshold', 'prod')
).reset_index()
    
# analysis - three tier melt
reddit_three = reddit_melted[reddit_melted['Tier']=='Three'].groupby(['Author', 'Bias']).agg(
    three_bias_count = ('Bias', 'count'),
    three_bias_prob = ('Threshold', 'prod')
).reset_index()

# analysis - both tiers melt
reddit_both = reddit_melted.groupby(['Author', 'Bias']).agg(
    bias_count = ('Bias', 'count'),
    bias_prob = ('Threshold', 'prod')
).reset_index()

# analysis - two tier pivot
reddit_two['Scaled'] = reddit_two['two_bias_count'] * reddit_two['two_bias_prob']
reddit_scaled = reddit_two[['Author', 'Bias', 'Scaled']].pivot(index='Author', columns='Bias', values='Scaled').reset_index()
reddit_scaled.fillna(0, inplace=True)
reddit_scaled['Sentiment Score'] = reddit_scaled['Left'] - reddit_scaled['Right']
score_bins = [-3, -1, 1, 3]
score_labels = ['Negative', 'Neutral', 'Positive']
reddit_scaled['Sentiment Label'] = pd.cut(reddit_scaled['Sentiment Score'], bins=score_bins, labels=score_labels, right=False)


# save analysis
reddit_scaled.to_csv('data/reddit_scaled.csv', index=False)


'''
PART 3: ANALYZE RESULTS - REDDIT USERS ILLUSTRATIONS
'''

# overall gauge plot
create_sentiment_gauge(reddit_scaled, 'images/sentiment_gauge_overall.html')

# subset by sentiment label
reddit_scaled_positive = reddit_scaled[reddit_scaled['Sentiment Label']=='Positive']
reddit_scaled_neutral = reddit_scaled[reddit_scaled['Sentiment Label']=='Neutral']
reddit_scaled_negative = reddit_scaled[reddit_scaled['Sentiment Label']=='Negative']

# sort by sentiment score
reddit_scaled_positive.sort_values('Sentiment Score', ascending=False, inplace=True)
reddit_scaled_neutral.sort_values('Sentiment Score', ascending=False, inplace=True)
reddit_scaled_negative.sort_values('Sentiment Score', ascending=True, inplace=True)

# reset indices
reddit_scaled_positive.reset_index(inplace=True, drop=True)
reddit_scaled_neutral.reset_index(inplace=True, drop=True)
reddit_scaled_negative.reset_index(inplace=True, drop=True)

# gauge plots by sentiment
create_sentiment_gauge(reddit_scaled_positive, 'images/sentiment_gauge_positive.html')
create_sentiment_gauge(reddit_scaled_neutral, 'images/sentiment_gauge_neutral.html')
create_sentiment_gauge(reddit_scaled_negative, 'images/sentiment_gauge_negative.html')

# get author content
reddit_content = reddit_author_reduced.copy()
reddit_content.drop(columns=['cleaned', 'length'], inplace=True)
reddit_content.rename(columns={'author': 'Author', 'author_content_aggregated': 'Content'}, inplace=True)

# merge content by sentiment
reddit_content_positive = pd.merge(reddit_scaled_positive, reddit_content, on='Author')
reddit_content_neutral = pd.merge(reddit_scaled_neutral, reddit_content, on='Author')
reddit_content_negative = pd.merge(reddit_scaled_negative, reddit_content, on='Author')

# save content
reddit_content_positive.to_csv('data/reddit_content_positive.csv', index=False)
reddit_content_neutral.to_csv('data/reddit_content_neutral.csv', index=False)
reddit_content_negative.to_csv('data/reddit_content_negative.csv', index=False)
