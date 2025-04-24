'''
Modeling Preparation
'''

# import functions from modeling_functions script
from modeling_functions import *

'''
PART 1: IMPORT AND PREPARE DATA
'''

# import newsapi data
newsapi_specific = pd.read_csv('../send_files/cleaning/newsapi_data_cleaned/newsapi_labeled_2_14_25.csv')
newsapi_training = pd.read_csv('../send_files/cleaning/newsapi_data_cleaned/training_labeled_2_14_25.csv')

# clean newsapi data
newsapi_specific['cleaned'] = newsapi_specific['article'].apply(preprocess)
newsapi_training['cleaned'] = newsapi_training['article'].apply(preprocess)

# combine newsapi dataframes
combine_columns = ['url', 'Bias Specific', 'cleaned']
newsapi_combined = pd.concat([newsapi_specific[combine_columns], newsapi_training[combine_columns]], ignore_index=True)

# drop any duplicates
newsapi_combined.drop_duplicates('url', inplace=True, ignore_index=True)

# save sample of newsapi data
newsapi_sample = newsapi_combined.sample(20, random_state=42, ignore_index=True).copy()
newsapi_sample = truncate_text(newsapi_sample, ['cleaned'], 50)
newsapi_sample.to_csv('data/newsapi_sample.csv', index=False)

# import reddit data
reddit_student_loan_forgiveness = pd.read_csv('../send_files/cleaning/reddit_data_cleaned/labeled_student_loan_forgiveness.csv')
reddit_student_loans = pd.read_csv('../send_files/cleaning/reddit_data_cleaned/labeled_student_loans.csv')

# add reddit labels for search query
reddit_student_loan_forgiveness['search'] = 'reddit_student_loan_forgiveness'
reddit_student_loans['search'] = 'reddit_student_loans'

# concatenate reddit data
reddit_labeled = pd.concat([reddit_student_loan_forgiveness, reddit_student_loans], ignore_index=True)

# clean reddit data
reddit_labeled['cleaned'] = reddit_labeled['author_content_aggregated'].apply(preprocess)
reddit_labeled.drop_duplicates(['author', 'cleaned'], inplace=True)

# aggregate reddit data by author
reddit_author = aggregate_into_schema(reddit_labeled, ['author', 'author_content_aggregated', 'cleaned'], 'author')

# remove AutoModerator
reddit_author = reddit_author[reddit_author['author'] != 'AutoModerator']
reddit_author.reset_index(drop=True, inplace=True)

# reduce reddit data to first quartile length of news articles
newsapi_combined['length'] = newsapi_combined['cleaned'].apply(lambda x: len(x.split()))
reddit_author['length'] = reddit_author['cleaned'].apply(lambda x: len(x.split()))
reddit_author_reduced = reddit_author[reddit_author['length'] >= newsapi_combined['length'].quantile(0.25)]

# create table to display the spreads - calculate statistics
newsapi_length_spread = newsapi_combined['length'].describe().reset_index()
reddit_length_spread = reddit_author['length'].describe().reset_index()
reddit_reduced_length_spread = reddit_author_reduced['length'].describe().reset_index()

# save reddit dataset
reddit_author_reduced.to_csv('data/reddit_author_reduced.csv', index=False)

# save sample of reddit author data
reddit_sample = reddit_author_reduced.sample(20, random_state=42, ignore_index=True).copy()
reddit_sample = truncate_text(reddit_sample, ['author_content_aggregated', 'cleaned'], 50)
reddit_sample.to_csv('data/reddit_sample.csv', index=False)

# create table to display the spreads - rename and combine statistics
newsapi_length_spread.columns = ['Statistic', 'News Article Value']
reddit_length_spread.columns = ['Statistic', 'Reddit Author Value']
reddit_reduced_length_spread.columns = ['Statistic', 'Reddit Subset Author Value']

# over length statistics
length_spreads = newsapi_length_spread.copy()
length_spreads = pd.merge(length_spreads, reddit_length_spread, on='Statistic')
length_spreads = pd.merge(length_spreads, reddit_reduced_length_spread, on='Statistic')

# save length spreads
length_spreads.to_csv('data/length_spreads.csv', index=False)

'''
PART 2: VECTORIZE THE DATA AND CREATE TRAIN/TEST SPLITS ON NEWSAPI DATA
'''

# countvectorize newsapi data
cv_params = {'stop_words': 'english',
             'max_features': 1000}
newaspi_cv = vectorize_to_df(newsapi_combined['cleaned'].tolist(), input_type='content', vectorizer_type='count', params=cv_params)

# label for newsapi data
labeled_newsapi_cv = pd.concat([newsapi_combined[['Bias Specific']], newaspi_cv], axis=1)
labeled_newsapi_cv.rename(columns={'Bias Specific': 'BIAS'}, inplace=True)

# newsapi 5 labels: left, lean-left, center, lean-right, right
newsapi_five = labeled_newsapi_cv.dropna()

# save smaple of newsapi_five
newsapi_cv_sample = newsapi_five.sample(20, random_state=23).copy()
newsapi_cv_sample.to_csv('data/newsapi_cv_sample.csv', index=False)

# newsapi 3 labels: lean-left and left combined, center, lean-right and right combined
newsapi_three = newsapi_five.replace({'Lean Left': 'Left', 'Lean Right': 'Right'})

# newsapi strict 3 labels: strictly left, strictly center, and strictly righty
newsapi_strict_three = newsapi_five[newsapi_five['BIAS'].isin(['Left', 'Center', 'Right'])]

# newsapi 2 labels: lean-left and left combined with lean-right and right combined
newsapi_two = newsapi_three[newsapi_three['BIAS'].isin(['Left', 'Right'])]

# newsapi strict 2 labels: strictly left and strictly right
newsapi_strict_two = newsapi_five[newsapi_five['BIAS'].isin(['Left', 'Right'])]

# modeling - five
X_five = newsapi_five.drop(columns='BIAS')
y_five = newsapi_five['BIAS']

# train test split - five
X_train_five, X_test_five, y_train_five, y_test_five = train_test_split(X_five, y_five, test_size=0.3, random_state=42)

# modeling - three
X_three = newsapi_three.drop(columns='BIAS')
y_three = newsapi_three['BIAS']

# train test split - three
X_train_three, X_test_three, y_train_three, y_test_three = train_test_split(X_three, y_three, test_size=0.3, random_state=42)

# modeling - strict three
X_strict_three = newsapi_strict_three.drop(columns='BIAS')
y_strict_three = newsapi_strict_three['BIAS']

# train test split - strict three
X_train_strict_three, X_test_strict_three, y_train_strict_three, y_test_strict_three = train_test_split(X_strict_three, y_strict_three, test_size=0.3, random_state=42)

# modeling - two
X_two = newsapi_strict_two.drop(columns='BIAS')
y_two = newsapi_strict_two['BIAS']

# train test split - two
X_train_two, X_test_two, y_train_two, y_test_two = train_test_split(X_two, y_two, test_size=0.3, random_state=42)

# modeling - strict two
X_strict_two = newsapi_strict_two.drop(columns='BIAS')
y_strict_two = newsapi_strict_two['BIAS']

# train test split - strict two
X_train_strict_two, X_test_strict_two, y_train_strict_two, y_test_strict_two = train_test_split(X_strict_two, y_strict_two, test_size=0.3, random_state=42)

# save the cv data
newsapi_five.to_csv('data/newsapi_five_cv.csv')
newsapi_three.to_csv('data/newsapi_three_cv.csv')
newsapi_strict_three.to_csv('data/newsapi_strict_three_cv.csv')
newsapi_two.to_csv('data/newsapi_two_cv.csv')
newsapi_strict_two.to_csv('data/newsapi_strict_two_cv.csv')

# save train/test split data heads - five
X_train_five.head(10).to_csv('data/train_test_split/X_train_five.csv')
X_test_five.head(10).to_csv('data/train_test_split/X_test_five.csv')
y_train_five.head(10).to_csv('data/train_test_split/y_train_five.csv')
y_test_five.head(10).to_csv('data/train_test_split/y_test_five.csv')

# save train/test split data heads - three
X_train_three.head(10).to_csv('data/train_test_split/X_train_three.csv')
X_test_three.head(10).to_csv('data/train_test_split/X_test_three.csv')
y_train_three.head(10).to_csv('data/train_test_split/y_train_three.csv')
y_test_three.head(10).to_csv('data/train_test_split/y_test_three.csv')

# save train/test split data heads - strict three
X_train_strict_three.head(10).to_csv('data/train_test_split/X_train_strict_three.csv')
X_test_strict_three.head(10).to_csv('data/train_test_split/X_test_strict_three.csv')
y_train_strict_three.head(10).to_csv('data/train_test_split/y_train_strict_three.csv')
y_test_strict_three.head(10).to_csv('data/train_test_split/y_test_strict_three.csv')

# save train/test split data heads - two
X_train_two.head(10).to_csv('data/train_test_split/X_train_two.csv')
X_test_two.head(10).to_csv('data/train_test_split/X_test_two.csv')
y_train_two.head(10).to_csv('data/train_test_split/y_train_two.csv')
y_test_two.head(10).to_csv('data/train_test_split/y_test_two.csv')

# save train/test split data heads - strict two
X_train_strict_two.head(10).to_csv('data/train_test_split/X_train_strict_two.csv')
X_test_strict_two.head(10).to_csv('data/train_test_split/X_test_strict_two.csv')
y_train_strict_two.head(10).to_csv('data/train_test_split/y_train_strict_two.csv')
y_test_strict_two.head(10).to_csv('data/train_test_split/y_test_strict_two.csv')


'''
PART 3: INVESTIGATE LABEL BALANCE
'''

# overall and training balance investigation - five
plot_overall_training(newsapi_five, y_train_five, ['Left', 'Lean Left', 'Center', 'Lean Right', 'Right'],
                      label_col='BIAS', title='Five', save_path='images/newsapi_proportions_five.png')

# overall and training balance investigation - three
plot_overall_training(newsapi_three, y_train_three, ['Left', 'Center', 'Right'],
                      label_col='BIAS', title='Three', save_path='images/newsapi_proportions_three.png')

# overall and training balance investigation - strict three
plot_overall_training(newsapi_strict_three, y_train_strict_three, ['Left', 'Center', 'Right'],
                      label_col='BIAS', title='Strict Three', save_path='images/newsapi_proportions_strict_three.png')

# overall and training balance investigation - two
plot_overall_training(newsapi_two, y_train_two, ['Left', 'Right'],
                      label_col='BIAS', title='Two', save_path='images/newsapi_proportions_two.png')

# overall and training balance investigation - strict two
plot_overall_training(newsapi_strict_two, y_train_strict_two, ['Left', 'Right'],
                      label_col='BIAS', title='Strict Two', save_path='images/newsapi_proportions_strict_two.png')


'''
PART 4: CREATE THE REDDIT DATA
'''
# count vectorizer created without maximum wordcount limit
cv_params = {'stop_words': 'english'}
reddit_author_cv = vectorize_to_df(reddit_author_reduced['cleaned'].tolist(), input_type='content', vectorizer_type='count', params=cv_params)

# save reddit cv
reddit_author_cv.to_csv('data/reddit_author_cv.csv', index=False)

# save reddit cv sample
reddit_author_cv.head(10).to_csv('data/reddit_author_cv_sample.csv', index=False)
