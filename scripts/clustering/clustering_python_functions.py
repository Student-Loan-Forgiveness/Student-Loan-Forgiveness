'''
Clustering Functions
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import os

# functin to remove outliers
def remove_outliers(cv_df, contamination=0.05):
    df_anomaly = cv_df.copy()
    iso_forest = IsolationForest(contamination=contamination)
    df_anomaly['ANOMALY'] = iso_forest.fit_predict(cv_df)
    # anomalies will be labeled as -1
    df_cleaned = df_anomaly[df_anomaly['ANOMALY']==1].copy().reset_index(drop=True)
    df_cleaned.drop('ANOMALY', axis=1, inplace=True)
    return df_cleaned

# function to perform 3D PCA
def run_pca(cv_df, title, save_path=False):
    scaler = StandardScaler()
    cv_scaled = scaler.fit_transform(cv_df)
    
    pca = PCA(n_components=3)
    pca_object = pca.fit_transform(cv_scaled)
    pca_df = pd.DataFrame(pca_object, columns=['component_1', 'component_2', 'component_3'])

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
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca_df

# function to visualize the 3D cluster results
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
    
    # begin figure
    fig = px.scatter_3d(df_results, x='component_1', y='component_2', z='component_3', color=label_col, opacity=opacity)
    
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
        title=legend_title,
        x=0.1,  # Position of the legend
        y=0.9,
        bgcolor='rgba(255, 255, 255, 0.5)',  # Background color with transparency
        bordercolor='black',
        borderwidth=2
    ))
    
    # write plotly figure to html
    fig.write_html(save_path)

# function to perform KMeans clustering
def kmeans_clustering(pca_df, min_clusters, max_clusters):
    results_df = pca_df.copy()
    
    for cluster in range(min_clusters, max_clusters + 1):
        model = KMeans(n_clusters=cluster)
        model_labels = model.fit_predict(pca_df)
        results_df[f'clusters_{cluster}'] = model_labels
        
    return results_df

# initial function to produce silhouette score results for the KMeans clustering
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
        
    return silhouette_df

# secondary function to produce silhouette average score results for the KMeans clustering
def plot_silhouette_averages(silhouette_concatenation, title, save_path=False):
    # groupby features into averages
    silhouette_averages = silhouette_concatenation.groupby('Features').mean().reset_index()
    
    # create plot
    plt.figure()
    sns.barplot(silhouette_averages, x='Features', y='scores', order=['Maximum', '200', '50'])
    plt.xlabel('Features')
    plt.ylabel('Average Silhouette Coefficient Score')
    plt.title(title)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
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

# see if sources were given the same clustering
def proportion_clusters_by_label(df, label_col, cluster_col):
    proportion_df = df.groupby([label_col, cluster_col]).size().groupby(level=0).apply(lambda x: x / x.sum()).reset_index(name='proportion')
    return proportion_df

# now we want to visualze the cluster majorities
def calculate_cluster_majorities(proportion_df, label_col):
    majority_cluster_df = proportion_df.loc[proportion_df.groupby(label_col)['proportion'].idxmax()].reset_index(drop=True)
    return majority_cluster_df