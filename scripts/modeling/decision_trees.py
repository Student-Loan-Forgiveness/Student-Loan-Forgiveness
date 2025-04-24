'''
Decision Tree Modeling
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
PART 3: DECISION TREE MODELING - GRID SEARCH

- Using a slightly different approach with grid-search cross-validation:
    - X, y set includes entirety of the rows
    - The grid-search aspect allows for hyper parameter tuning
    - The cross-validation process internally creates different disjoint train/test subsets
'''

# decision tree parameters to test
param_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
              'max_depth': [None, 5, 10],
              'max_features': [None, 'sqrt', 'log2']}

# labels and data
labels = ['Five', 'Three', 'Strict Three', 'Two', 'Strict Two']
X_data = [X_five, X_three, X_strict_three, X_two, X_strict_two]
y_data = [y_five, y_three, y_strict_three, y_two, y_strict_two]

# run grid search over the hyperparameters, labels, and data
results_df, results_detailed = iterate_grid_search(DecisionTreeClassifier(random_state=42), param_grid, labels, X_data, y_data, return_detail=True)

# save results
results_df.to_csv('grid_search.csv', index=False)

# extract the detailed results
detailed_five = pd.DataFrame(results_detailed['Five'])
detailed_three = pd.DataFrame(results_detailed['Three'])
detailed_strict_three = pd.DataFrame(results_detailed['Strict Three'])
detailed_two = pd.DataFrame(results_detailed['Two'])
detailed_strict_two = pd.DataFrame(results_detailed['Strict Two'])

# save the detailed results
detailed_five.to_csv('detailed_five.csv', index=False)
detailed_three.to_csv('detailed_three.csv', index=False)
detailed_strict_three.to_csv('detailed_strict_three.csv', index=False)
detailed_two.to_csv('detailed_two.csv', index=False)
detailed_strict_two.to_csv('detailed_strict_two.csv', index=False)

# combine the detailed results
combined_details = combine_iterated_details(results_detailed)

# save combined details
combined_details.to_csv('detailed_combined.csv', index=False)


'''
PART 4: DECISION TREE MODELING
'''

# dt - five labels
model_five = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model_five.fit(X_train_five, y_train_five)
y_pred_five = model_five.predict(X_test_five)
accuracy_five = accuracy_score(y_test_five, y_pred_five)
cm_five = confusion_matrix(y_test_five, y_pred_five)
illustrate_cm(accuracy_five, cm_five, model_classes=model_five.classes_.tolist(),
              classes_order=['Left', 'Lean Left', 'Center', 'Lean Right', 'Right'],
              title='Five Labels', save_path='cm_five_labels.png')
illustrate_dt(model_five, X_train_five, accuracy_five, title='Five Labels', save_path='dt_five_labels.png')

# dt - three labels
model_three = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
model_three.fit(X_train_three, y_train_three)
y_pred_three = model_three.predict(X_test_three)
accuracy_three = accuracy_score(y_test_three, y_pred_three)
cm_three = confusion_matrix(y_test_three, y_pred_three)
illustrate_cm(accuracy_three, cm_three, model_classes=model_three.classes_.tolist(),
              classes_order=['Left', 'Center', 'Right'],
              title='Three Labels', save_path='cm_three_labels.png')
illustrate_dt(model_three, X_train_three, accuracy_three, title='Three Labels', save_path='dt_three_labels.png')

# dt - strict three labels
model_strict_three = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
model_strict_three.fit(X_train_strict_three, y_train_strict_three)
y_pred_strict_three = model_strict_three.predict(X_test_strict_three)
accuracy_strict_three = accuracy_score(y_test_strict_three, y_pred_strict_three)
cm_strict_three = confusion_matrix(y_test_strict_three, y_pred_strict_three)
illustrate_cm(accuracy_strict_three, cm_strict_three, model_classes=model_strict_three.classes_.tolist(),
              classes_order=['Left', 'Center', 'Right'],
              title='Strict Three Labels', save_path='cm_strict_three_labels.png')
illustrate_dt(model_strict_three, X_train_strict_three, accuracy_strict_three, title='Strict Three Labels', save_path='dt_strict_three_labels.png')

# dt - two labels
model_two = DecisionTreeClassifier(criterion='gini', max_depth=5, max_features='log2', random_state=42)
model_two.fit(X_train_two, y_train_two)
y_pred_two = model_two.predict(X_test_two)
accuracy_two = accuracy_score(y_test_two, y_pred_two)
cm_two = confusion_matrix(y_test_two, y_pred_two)
illustrate_cm(accuracy_two, cm_two, model_classes=model_two.classes_.tolist(),
              classes_order=['Left', 'Right'],
              title='Two Labels', save_path='cm_two_labels.png')
illustrate_dt(model_two, X_train_two, accuracy_two, title='Two Labels', save_path='dt_two_labels.png')

# dt - strict two labels
model_strict_two = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model_strict_two.fit(X_train_strict_two, y_train_strict_two)
y_pred_strict_two = model_two.predict(X_test_strict_two)
accuracy_strict_two = accuracy_score(y_test_strict_two, y_pred_strict_two)
cm_strict_two = confusion_matrix(y_test_strict_two, y_pred_strict_two)
illustrate_cm(accuracy_strict_two, cm_strict_two, model_classes=model_strict_two.classes_.tolist(),
              classes_order=['Left', 'Right'],
              title='Strict Two Labels', save_path='cm_strict_two_labels.png')
illustrate_dt(model_strict_two, X_train_strict_two, accuracy_strict_two, title='Strict Two Labels', save_path='dt_strict_two_labels.png')


'''
PART 5: REDUCE WITH FEATURE IMPORTANCE

Models:
    - Strict Two Labels: 80.25% Accuracy
'''

# calculate feature importances
feature_importance_two = calculate_feature_importance(model_strict_two, X_test_strict_two, y_test_strict_two)

# save feature importances
feature_importance_two.to_csv('feature_importance_two.csv', index=False)

# also include any nodes not within important features
node_features_two = list(set(model_strict_two.feature_names_in_[model_strict_two.tree_.feature].tolist()))

# subset the train / test data - two classes
important_features_two = feature_importance_two[feature_importance_two['absolute_importance'] > 0]['feature'].tolist()
important_features_two = list(set(node_features_two).union(set(important_features_two)))
X_train_important_two = X_train_strict_two[important_features_two]
X_test_important_two = X_test_strict_two[important_features_two]
y_train_important_two = y_train_strict_two.copy()
y_test_important_two = y_test_strict_two.copy()


'''
PART 6: RERUN WITH FEATURE IMPORTANCE

Models:
    - Strict Two Labels: 80.25% Accuracy
'''

# dt - important two classes
model_important_two = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model_important_two.fit(X_train_important_two, y_train_important_two)
y_pred_important_two = model_important_two.predict(X_test_important_two)
accuracy_important_two = accuracy_score(y_test_important_two, y_pred_important_two)
cm_important_two = confusion_matrix(y_test_important_two, y_pred_important_two)
illustrate_cm(accuracy_important_two, cm_important_two, model_classes=model_important_two.classes_.tolist(),
              classes_order=['Left', 'Right'],
              title='Important Two Labels', save_path='cm_important_two_labels.png')
illustrate_dt(model_important_two, X_train_important_two, accuracy_important_two, title='Important Two Labels', save_path='dt_important_two_labels.png')

# save the model - 2 labels
with open('dt_two.pkl', 'wb') as file:
    pickle.dump(model_important_two, file)

# to load the model - 2 labels
with open('dt_two.pkl', 'rb') as file:
    model_important_two = pickle.load(file)


'''
PART 7: APPLY FEATURES TO THE REDDIT COUNTVECTORIZED DATA

REDDIT DATA IS UNLABELED:
    - the reddit data is unlabeled like the testing data
    - it's using the better performing models to project the label onto the reddit author
'''

# populate and reduce to important features
reddit_two = populate_important_features(reddit_cv, important_features_two)

# save data
reddit_two.to_csv('reddit_two_cv.csv', index=False)

# save samples
reddit_two.head(10).to_csv('reddit_two_cv_sample.csv', index=False)

# make predictions and return probabilities - 2 labels
reddit_two_predictions = model_important_two.predict(reddit_two)
reddit_two_probs = model_important_two.predict_proba(reddit_two)


'''
PART 8: ANALYZE REDDIT APPLICATION
'''

# look at probability spreads - 2
reddit_results_two = pd.DataFrame(reddit_two_probs, columns=model_important_two.classes_)
reddit_results_two['Predicted Bias'] = reddit_two_predictions
reddit_results_two['Author'] = reddit_author_reduced['author']
reddit_results_two['Threshold'] = reddit_results_two.apply(lambda row: max(row['Left'], row['Right']), axis=1)

# drop and reorder
reorder_columns = ['Author', 'Predicted Bias', 'Threshold']
reddit_results_two = reddit_results_two[reorder_columns]

# save results
reddit_results_two.to_csv('reddit_projection_two.csv', index=False)

# load results
reddit_results_two = pd.read_csv('reddit_projection_two.csv')

# copy and alter results
reddit_results = reddit_results_two.copy()
reddit_results.rename(columns={'Predicted Bias': 'Predicted Bias Two',
                               'Threshold': 'Threshold Two'}, inplace=True)
reddit_results['Threshold'] = reddit_results['Threshold Two']
reddit_results['Conclusion'] = reddit_results['Predicted Bias Two']
        
# save results
reddit_results.to_csv('dt_reddit_results.csv', index=False)
        