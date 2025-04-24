'''
Naive Bayes Modeling
'''

# import modeling functions
import sys
sys.path.append('../')
from modeling_functions import *

# import libraries not within modeling_functions
import pickle


'''
PART 1: IMPORT DATA
'''

# import data - newsapi countvectorized dataframes
newsapi_five = pd.read_csv('../data/newsapi_five_cv.csv', index_col='Unnamed: 0')
newsapi_three = pd.read_csv('../data/newsapi_three_cv.csv', index_col='Unnamed: 0')
newsapi_strict_three = pd.read_csv('../data/newsapi_strict_three_cv.csv', index_col='Unnamed: 0')
newsapi_two = pd.read_csv('../data/newsapi_two_cv.csv', index_col='Unnamed: 0')
newsapi_strict_two = pd.read_csv('../data/newsapi_strict_two_cv.csv', index_col='Unnamed: 0')

# import data - reddit author aggregation schema and countvectorized dataframe
reddit_author_reduced = pd.read_csv('../data/reddit_author_reduced.csv')
reddit_cv = pd.read_csv('../data/reddit_author_cv.csv')


'''
PART 2: CREATING TRAIN / TEST SPLIT DATA
'''

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
X_two = newsapi_two.drop(columns='BIAS')
y_two = newsapi_two['BIAS']

# train test split - two
X_train_two, X_test_two, y_train_two, y_test_two = train_test_split(X_two, y_two, test_size=0.3, random_state=42)

# modeling - strict two
X_strict_two = newsapi_strict_two.drop(columns='BIAS')
y_strict_two = newsapi_strict_two['BIAS']

# train test split - strict two
X_train_strict_two, X_test_strict_two, y_train_strict_two, y_test_strict_two = train_test_split(X_strict_two, y_strict_two, test_size=0.3, random_state=42)


'''
PART 3: NAIVE BAYES MODELING
'''

# nb - five labels
model_five = MultinomialNB()
model_five.fit(X_train_five, y_train_five)
y_pred_five = model_five.predict(X_test_five)
accuracy_five = accuracy_score(y_test_five, y_pred_five)
cm_five = confusion_matrix(y_test_five, y_pred_five)
illustrate_cm(accuracy_five, cm_five, model_classes=model_five.classes_.tolist(),
              classes_order=['Left', 'Lean Left', 'Center', 'Lean Right', 'Right'],
              title='Five Labels', save_path='cm_five_labels.png')

# nb - three labels
model_three = MultinomialNB()
model_three.fit(X_train_three, y_train_three)
y_pred_three = model_three.predict(X_test_three)
accuracy_three = accuracy_score(y_test_three, y_pred_three)
cm_three = confusion_matrix(y_test_three, y_pred_three)
illustrate_cm(accuracy_three, cm_three, model_classes=model_three.classes_.tolist(),
              classes_order=['Left', 'Center', 'Right'],
              title='Three Labels', save_path='cm_three_labels.png')

# nb - strict three labels
model_strict_three = MultinomialNB()
model_strict_three.fit(X_train_strict_three, y_train_strict_three)
y_pred_strict_three = model_strict_three.predict(X_test_strict_three)
accuracy_strict_three = accuracy_score(y_test_strict_three, y_pred_strict_three)
cm_strict_three = confusion_matrix(y_test_strict_three, y_pred_strict_three)
illustrate_cm(accuracy_strict_three, cm_strict_three, model_classes=model_strict_three.classes_.tolist(),
              classes_order=['Left', 'Center', 'Right'],
              title='Strict Three Labels', save_path='cm_strict_three_labels.png')

# nb - two labels
model_two = MultinomialNB()
model_two.fit(X_train_two, y_train_two)
y_pred_two = model_two.predict(X_test_two)
accuracy_two = accuracy_score(y_test_two, y_pred_two)
cm_two = confusion_matrix(y_test_two, y_pred_two)
illustrate_cm(accuracy_two, cm_two, model_classes=model_two.classes_.tolist(),
              classes_order=['Left', 'Right'],
              title='Two Labels', save_path='cm_two_labels.png')

# nb - strict two labels
model_strict_two = MultinomialNB()
model_strict_two.fit(X_train_strict_two, y_train_strict_two)
y_pred_strict_two = model_two.predict(X_test_strict_two)
accuracy_strict_two = accuracy_score(y_test_strict_two, y_pred_strict_two)
cm_strict_two = confusion_matrix(y_test_strict_two, y_pred_strict_two)
illustrate_cm(accuracy_strict_two, cm_strict_two, model_classes=model_strict_two.classes_.tolist(),
              classes_order=['Left', 'Right'],
              title='Strict Two Labels', save_path='cm_strict_two_labels.png')


'''
PART 4: REDUCE WITH FEATURE IMPORTANCE

Models:
    - Strict Three Labels: 75.69% Accuracy
    - Strict Two Labels: 83.95% Accuracy
'''

# calculate feature importances
feature_importance_three = calculate_feature_importance(model_strict_three, X_test_strict_three, y_test_strict_three)
feature_importance_two = calculate_feature_importance(model_strict_two, X_test_strict_two, y_test_strict_two)

# save feature importances
feature_importance_three.to_csv('feature_importance_three.csv', index=False)
feature_importance_two.to_csv('feature_importance_two.csv', index=False)

# subset the train / test data - three classes
important_features_three = feature_importance_three[feature_importance_three['absolute_importance'] > 0]['feature'].tolist()
X_train_important_three = X_train_strict_three[important_features_three]
X_test_important_three = X_test_strict_three[important_features_three]
y_train_important_three = y_train_strict_three.copy()
y_test_important_three = y_test_strict_three.copy()

# subset the train / test data - two classes
important_features_two = feature_importance_two[feature_importance_two['absolute_importance'] > 0]['feature'].tolist()
X_train_important_two = X_train_strict_two[important_features_two]
X_test_important_two = X_test_strict_two[important_features_two]
y_train_important_two = y_train_strict_two.copy()
y_test_important_two = y_test_strict_two.copy()


'''
PART 5: RERUN WITH FEATURE IMPORTANCE

Models:
    - Strict Three Labels: 75.69% Accuracy
    - Strict Two Labels: 83.95% Accuracy
'''

# nb - important three classes
model_important_three = MultinomialNB()
model_important_three.fit(X_train_important_three, y_train_important_three)
y_pred_important_three = model_important_three.predict(X_test_important_three)
accuracy_important_three = accuracy_score(y_test_important_three, y_pred_important_three)
cm_important_three = confusion_matrix(y_test_important_three, y_pred_important_three)
illustrate_cm(accuracy_important_three, cm_important_three, model_classes=model_important_three.classes_.tolist(),
              classes_order=['Left', 'Center', 'Right'],
              title='Important Three Labels', save_path='cm_important_three_labels.png')

# nb - important two classes
model_important_two = MultinomialNB()
model_important_two.fit(X_train_important_two, y_train_important_two)
y_pred_important_two = model_important_two.predict(X_test_important_two)
accuracy_important_two = accuracy_score(y_test_important_two, y_pred_important_two)
cm_important_two = confusion_matrix(y_test_important_two, y_pred_important_two)
illustrate_cm(accuracy_important_two, cm_important_two, model_classes=model_important_two.classes_.tolist(),
              classes_order=['Left', 'Right'],
              title='Important Two Labels', save_path='cm_important_two_labels.png')

# save the models - 3 labels
with open('nb_three.pkl', 'wb') as file:
    pickle.dump(model_important_three, file)

# save the models - 2 labels
with open('nb_two.pkl', 'wb') as file:
    pickle.dump(model_important_two, file)
    
# to load the models - 3 labels
with open('nb_three.pkl', 'rb') as file:
    model_important_three = pickle.load(file)

# to load the models - 2 labels
with open('nb_two.pkl', 'rb') as file:
    model_important_two = pickle.load(file)


'''
PART 6: APPLY FEATURES TO THE REDDIT COUNTVECTORIZED DATA

REDDIT DATA IS UNLABELED:
    - the reddit data is unlabeled like the testing data
    - it's using the better performing models to project the label onto the reddit author
'''

# populate and reduce to important features
reddit_three = populate_important_features(reddit_cv, important_features_three)
reddit_two = populate_important_features(reddit_cv, important_features_two)

# save data
reddit_three.to_csv('reddit_three_cv.csv', index=False)
reddit_two.to_csv('reddit_two_cv.csv', index=False)

# save samples
reddit_three.head(10).to_csv('reddit_three_cv_sample.csv', index=False)
reddit_two.head(10).to_csv('reddit_two_cv_sample.csv', index=False)

# make predictions and return probabilities - 3 labels
reddit_three_predictions = model_important_three.predict(reddit_three)
reddit_three_probs = model_important_three.predict_proba(reddit_three)

# make predictions and return probabilities - 2 labels
reddit_two_predictions = model_important_two.predict(reddit_two)
reddit_two_probs = model_important_two.predict_proba(reddit_two)


'''
PART 7: ANALYZE REDDIT APPLICATION
'''

# look at probability spreads - 3
reddit_results_three = pd.DataFrame(reddit_three_probs, columns=model_important_three.classes_)
reddit_results_three['Predicted Bias'] = reddit_three_predictions
reddit_results_three['Author'] = reddit_author_reduced['author']
reddit_results_three['Threshold'] = reddit_results_three.apply(lambda row: max(row['Left'], row['Center'], row['Right']), axis=1)

# drop and reorder
reorder_columns = ['Author', 'Predicted Bias', 'Threshold']
reddit_results_three = reddit_results_three[reorder_columns]

# look at probability spreads - 2
reddit_results_two = pd.DataFrame(reddit_two_probs, columns=model_important_two.classes_)
reddit_results_two['Predicted Bias'] = reddit_two_predictions
reddit_results_two['Author'] = reddit_author_reduced['author']
reddit_results_two['Threshold'] = reddit_results_two.apply(lambda row: max(row['Left'], row['Right']), axis=1)

# drop and reorder
reorder_columns = ['Author', 'Predicted Bias', 'Threshold']
reddit_results_two = reddit_results_two[reorder_columns]

# save results
reddit_results_three.to_csv('reddit_projection_three.csv', index=False)
reddit_results_two.to_csv('reddit_projection_two.csv', index=False)

# load results
reddit_results_three = pd.read_csv('reddit_projection_three.csv')
reddit_results_two = pd.read_csv('reddit_projection_two.csv')

# combine reddit results
reddit_results = pd.merge(reddit_results_three, reddit_results_two, on='Author', suffixes=(' Three', ' Two'))
reddit_results['Threshold'] = reddit_results['Threshold Three'] * reddit_results['Threshold Two']

# conclusion on political bias
reddit_results['Conclusion'] = None
for index, row in reddit_results.iterrows():
    bias_three = row['Predicted Bias Three']
    bias_two = row['Predicted Bias Two']
    if bias_three == bias_two:
        reddit_results.loc[index, 'Conclusion'] = bias_three
    elif bias_three == 'Center':
        reddit_results.loc[index, 'Conclusion'] = f'Lean {bias_two}'
    else:
        reddit_results.loc[index, 'Conclusion'] = 'Center'
        
# save results
reddit_results.to_csv('nb_reddit_results.csv', index=False)
