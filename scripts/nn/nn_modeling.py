'''
Modeling with Neural Network Class
'''

# import sklearn libraries
from sklearn.model_selection import train_test_split

# import custom functions (includes libraries)
from nn_functions import *
from modeling_functions import *


'''
Applying to Text Data
'''

# import newsapi data for text summaries
newsapi_specific = pd.read_csv('../send_files/cleaning/newsapi_data_cleaned/newsapi_labeled_2_14_25.csv')
newsapi_general = pd.read_csv('../send_files/cleaning/newsapi_data_cleaned/training_labeled_2_14_25.csv')

# combine newsapi dataframes
combine_columns = ['url', 'Bias Specific', 'article']
df = pd.concat([newsapi_specific[combine_columns], newsapi_general[combine_columns]], ignore_index=True)

# drop any duplicates
df.drop_duplicates('url', inplace=True, ignore_index=True)

# clean newsapi data
df['cleaned'] = df['article'].apply(preprocess)

# save a sample of this data
df.sample(20).to_csv('data/newsapi_sample.csv', index=False)

# countvectorize newsapi data
cv_params = {'stop_words': 'english',
             'max_features': 1000}
df_cv = vectorize_to_df(df['cleaned'].tolist(), input_type='content', vectorizer_type='count', params=cv_params)

# label for newsapi data
labeled_df = pd.concat([df[['Bias Specific']], df_cv], axis=1)
labeled_df.rename(columns={'Bias Specific': 'BIAS'}, inplace=True)

# reduce to left and right political biases
df_binary = labeled_df[labeled_df['BIAS'].isin(['Left', 'Right'])]

# create numeric representations of the binary labels
df_binary['BIAS'].replace({'Left': 1, 'Right': 0}, inplace=True)

# reset indices
df_binary.reset_index(drop=True, inplace=True)

# save binary cv
df_binary.to_csv('data/binary_cv.csv', index=False)

# scale and run pca
pca_df = run_pca(df_binary.drop(columns='BIAS'), 'Bias PCA', n_components=10, save_path='images/pca_ten.png')

# X and Y with binary labels and 10 component PCA
# first - train test split
X_train, X_test, y_train, y_test = train_test_split(pca_df, df_binary['BIAS'], test_size=0.2)

# save samples of train test split
X_train.to_csv('data/X_train_pca.csv')
X_test.to_csv('data/X_test_pca.csv')
y_train.to_csv('data/y_train_pca.csv')
y_test.to_csv('data/y_test_pca.csv')

# second - create numpy arrays
X_train_np = np.array(X_train)
X_test_np = np.array(X_test)
y_train_np = np.array(y_train).reshape(1, len(y_train))
y_test_np = np.array(y_test)

# function to gridsearch test architectures
def gridsearch_architectures(architecture_schemas, X_train, X_test, y_train, y_test, epochs=10000, alpha=0.001, random_state=None):
    # results data structure
    results = {'Architecture': [],
               'Train Accuracy': [],
               'Train Cost': [],
               'Train Time': [],
               'Test Accuracy': [],
               'Test F1-Score': []}

    for schema in architecture_schemas:
        # train the model
        schema_w, schema_b, schema_m = train(X_train, y_train,
                                             architecture_schemas[schema],
                                             epochs=epochs, alpha=alpha, random_state=random_state)
        # get the training metrics
        results['Architecture'].append(schema)
        results['Train Accuracy'].append(schema_m['accuracy'])
        results['Train Cost'].append(schema_m['cost'])
        results['Train Time'].append(schema_m['elapsed'])
        
        # test the model
        predictions, raw_predictions = predict(X_test, schema_w, schema_b)

        # accuracy
        test_accuracy = accuracy_score(y_test, predictions)

        # f1-score
        test_f1 = f1_score(y_test, predictions)
        
        # get the testing metrics
        results['Test Accuracy'].append(test_accuracy)
        results['Test F1-Score'].append(test_f1)
        
    # to dataframe
    results_df = pd.DataFrame(results)

    return results_df

pca_architecture_schemas = {'pca_architecture_1': [10, 16, 1],
                            'pca_architecture_2': [10, 32, 1],
                            'pca_architecture_3': [10, 64, 1],
                            'pca_architecture_4': [10, 128, 1],
                            'pca_architecture_5': [10, 16, 16, 1],
                            'pca_architecture_6': [10, 32, 32, 1],
                            'pca_architecture_7': [10, 64, 64, 1],
                            'pca_architecture_8': [10, 128, 128, 1],
                            'pca_architecture_9': [10, 16, 16, 16, 1],
                            'pca_architecture_10': [10, 32, 32, 32, 1],
                            'pca_architecture_11': [10, 64, 64, 64, 1],
                            'pca_architecture_12': [10, 128, 128, 128, 1]}

# initial run
initial_run = gridsearch_architectures(pca_architecture_schemas,
                                       X_train_np, X_test_np, y_train_np, y_test_np,
                                       epochs=10000, alpha=0.001, random_state=42)

# save initial results
initial_run.to_csv('data/nn_initial_run.csv', index=False)


# best model - train
best_w, best_b, best_m = train(X_train_np, y_train_np,
                               pca_architecture_schemas['pca_architecture_5'],
                               epochs=100000, alpha=0.001, random_state=42)

# best model - test
predictions, raw_predictions = predict(X_test_np, best_w, best_b)

# accuracy
best_accuracy = accuracy_score(y_test_np, predictions)

# f1-score
best_f1 = f1_score(y_test_np, predictions)

# cm
cm = confusion_matrix(y_test_np, predictions)
illustrate_cm(best_accuracy, cm, [0, 1], label_map={0: 'Right', 1: 'Left'}, classes_order=['Left', 'Right'],
              title='Bespoke Simple NN', save_path='images/bespoke_simple_nn_cm.png')

# create dataframe for plotting
best_df = pd.DataFrame({'Accuracy': best_m['accuracies'],
                       'Cost': best_m['costs'],
                       'Split': best_m['splits']})

# add epochs column
best_df['Epoch'] = best_df.index + 1

# reformat accuracies
best_df['Accuracy'] = best_df['Accuracy'] / 100

# melt the dataframe
best_melted = best_df[['Accuracy', 'Cost', 'Epoch']].melt(id_vars='Epoch', var_name='Metric', value_name='Score')

# plot
sns.lineplot(best_melted, x='Epoch', y='Score', hue='Metric')
plt.axvline(x=10000, color='darkgray', linestyle='--', label='Original Training Epochs')
plt.plot([0, 100000], [0.72, 0.72], color='midnightblue', linestyle=(0, (5, 10)), label='Original Testing Accuracy')
plt.plot([0, 100000], [0.70, 0.70], color='darkred', linestyle=(0, (5, 10)), label='Extended Testing Accuracy')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Bespoke Simple NN Extended Training Results')
plt.savefig('images/bespoke_simple_nn_metrics.png', dpi=300, bbox_inches='tight')
plt.show()


'''
Run on the Full 1000 word CV data
'''
# X and Y with binary labels and 1000-word CV dataframe
# first - train test split
X_train, X_test, y_train, y_test = train_test_split(df_binary.drop(columns=['BIAS']), df_binary['BIAS'], test_size=0.2)

# save samples of train test split
X_train.to_csv('data/X_train_full.csv')
X_test.to_csv('data/X_test_full.csv')
y_train.to_csv('data/y_train_full.csv')
y_test.to_csv('data/y_test_full.csv')

# second - create numpy arrays
X_train_np = np.array(X_train)
X_test_np = np.array(X_test)
y_train_np = np.array(y_train).reshape(1, len(y_train))
y_test_np = np.array(y_test)

full_architecture_schemas = {'full_architecture_1': [1000, 16, 1],
                             'full_architecture_2': [1000, 32, 1],
                             'full_architecture_3': [1000, 64, 1],
                             'full_architecture_4': [1000, 128, 1],
                             'full_architecture_5': [1000, 16, 16, 1],
                             'full_architecture_6': [1000, 32, 32, 1],
                             'full_architecture_7': [1000, 64, 64, 1],
                             'full_architecture_8': [1000, 128, 128, 1],
                             'full_architecture_9': [1000, 16, 16, 16, 1],
                             'full_architecture_10': [1000, 32, 32, 32, 1],
                             'full_architecture_11': [1000, 64, 64, 64, 1],
                             'full_architecture_12': [1000, 128, 128, 128, 1]}

# initial run
full_run = gridsearch_architectures(full_architecture_schemas,
                                    X_train_np, X_test_np, y_train_np, y_test_np,
                                    epochs=10000, alpha=0.001, random_state=42)

# save initial results
full_run.to_csv('data/nn_full_run.csv', index=False)

# best model - train
best_w, best_b, best_m = train(X_train_np, y_train_np,
                               full_architecture_schemas['full_architecture_1'],
                               epochs=100000, alpha=0.001, random_state=42)

# best model - test
predictions, raw_predictions = predict(X_test_np, best_w, best_b)

# accuracy
best_accuracy = accuracy_score(y_test_np, predictions)

# f1-score
best_f1 = f1_score(y_test_np, predictions)

# cm
cm = confusion_matrix(y_test_np, predictions)
illustrate_cm(best_accuracy, cm, [0, 1], label_map={0: 'Right', 1: 'Left'}, classes_order=['Left', 'Right'],
              title='Bespoke Large NN', save_path='images/bespoke_large_nn_cm.png')

# create dataframe for plotting
best_df = pd.DataFrame({'Accuracy': best_m['accuracies'],
                       'Cost': best_m['costs'],
                       'Split': best_m['splits']})

# add epochs column
best_df['Epoch'] = best_df.index + 1

# reformat accuracies
best_df['Accuracy'] = best_df['Accuracy'] / 100

# melt the dataframe
best_melted = best_df[['Accuracy', 'Cost', 'Epoch']].melt(id_vars='Epoch', var_name='Metric', value_name='Score')

# plot
sns.lineplot(best_melted, x='Epoch', y='Score', hue='Metric')
plt.axvline(x=10000, color='darkgray', linestyle='--', label='Original Training Epochs')
plt.plot([0, 100000], [0.90, 0.90], color='midnightblue', linestyle=(0, (5, 10)), label='Original Testing Accuracy')
plt.plot([0, 100000], [0.87, 0.87], color='darkred', linestyle=(0, (5, 10)), label='Extended Testing Accuracy')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Bespoke Large NN Extended Training Results')
plt.savefig('images/bespoke_large_nn_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# best model - train
best_w, best_b, best_m = train(X_train_np, y_train_np,
                               full_architecture_schemas['full_architecture_1'],
                               epochs=20000, alpha=0.001, random_state=42)

# best model - test
predictions, raw_predictions = predict(X_test_np, best_w, best_b)

# accuracy
best_accuracy = accuracy_score(y_test_np, predictions)

# f1-score
best_f1 = f1_score(y_test_np, predictions)

# cm
cm = confusion_matrix(y_test_np, predictions)
illustrate_cm(best_accuracy, cm, [0, 1], label_map={0: 'Right', 1: 'Left'}, classes_order=['Left', 'Right'],
              title='Bespoke Large NN Tuned', save_path='images/bespoke_large_nn_tuned_cm.png')


'''
# HELPER FUNCTIONS AND PROCESSES

# function to create sample data for testing
def create_sample_data(architecture):
    # test data (m x n) - (rows x features)
    X = np.array([
        [150, 70],
        [254, 73],
        [312, 68],
        [120, 60],
        [154, 61],
        [212, 65],
        [216, 67],
        [145, 67],
        [184, 64],
        [130, 69]])

    # test labels (1 x m)  - (output_nodes x rows)
    y = np.array([
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        0])

    # reshape for (output_nodes x rows) format
    Y = y.reshape(architecture[-1], len(y))
    
    return X, Y

# architecture
architecture = [10, 16, 16, 1]

# create layers
layers_df = create_layers(architecture)

# create dense
edges_df = create_dense(layers_df)

# create network
G = create_network(layers_df, edges_df, node_labels=False, edge_labels=False, node_size=10, save_path=None)
'''