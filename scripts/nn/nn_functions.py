'''
Neural Network Functions
'''

# import libraries
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time

# function to create layer architecture
def create_layers(architecture):
    '''
    Function to represent the layers of a neural network with a dataframe.

    Parameters
    ----------
    architecture : list of integers
        Number of nodes each layer of the neural network. Order matters.
        Example: architecture = [2, 3, 3, 1]
            - Input Layer: 2 nodes
            - Hidden Layer 1: 3 nodes
            - Hidden Layer 2: 3 nodes
            - Output Layer: 1 node
        

    Returns
    -------
    layers_df : pandas dataframe
        Pandas dataframe with the nodes, layers, and positions.
        Positions have symmetric y-positions.

    '''
    # initialize dictionary for structure storage
    structure = {'node': [],
                 'layer': [],
                 'pos': []}

    # iterate through node counts
    for layer_num, layer_count in enumerate(architecture):
        # symmetric y-positions
        y_positions = [(node - (layer_count - 1) / 2) for node in range(layer_count)]
        for node, y in enumerate(y_positions):
            structure['node'].append(f'node_{layer_num}_{node}')
            structure['layer'].append(layer_num)
            structure['pos'].append((layer_num, y))
                
    # create dataframe
    layers_df = pd.DataFrame(structure)

    return layers_df

# function to create dense edges
def create_dense(layers_df):
    '''
    Funce to create connection mappings for a fully-connected (dense) edges
    in a neural network.

    Parameters
    ----------
    layers_df : pandas dataframe
        Result from create_layers().

    Returns
    -------
    edges_df : pandas dataframe
        Fully-connected (dense) source and target connections
        for the layer_df neural network.

    '''
    # initialize dictionary for edge list
    edges = {'source': [],
             'target': []}
    
    # get the layer counts
    max_layer = layers_df['layer'].max()
    
    # iterate backwards through the layers
    for layer in range(max_layer, 0, -1):
        # get source and target nodes
        source_nodes = layers_df[layers_df['layer']==layer-1]['node'].tolist()
        target_nodes = layers_df[layers_df['layer']==layer]['node'].tolist()
        
        # create edges between source and target nodes
        for source in source_nodes:
            for target in target_nodes:
                edges['source'].append(source)
                edges['target'].append(target)
                
    # create dataframe
    edges_df = pd.DataFrame(edges)

    return edges_df

# function to create network object
def create_network(layers_df, edges_df, create_plot=True, save_path=None, node_size=1000, node_color='lightblue', node_labels=False, edge_labels=False):
    '''
    Function to create network object of neural network, options to plot and save.

    Parameters
    ----------
    layers_df : pandas dataframe
        DESCRIPTION.
    edges_df : pandas dataframe
        DESCRIPTION.
    create_plot : boolean, optional
        Plot or not. The default is True.
    save_path : string, optional
        Save path for network plot. Please specify file type. The default is False.
    node_size: integer, optional
        Size of the nodes. The default is 1000.
    node_color: string, optional
        Color of the nodes. The dafulat is "lightblue".

    Returns
    -------
    G : networkx digraph
        Directed Graph object of the Neural Network.

    '''
    # initialize directed graph
    G = nx.DiGraph()
    
    # add nodes and positions
    for index, row in layers_df.iterrows():
        G.add_node(row['node'], pos=row['pos'])
        
    # dictionary for edge labels
    edge_label_dict = {}
    
    # add edges
    for index, row in edges_df.iterrows():
        weight_label = f'w_{row["source"].split("_")[1]}_{row["source"].split("_")[2]}'
        G.add_edge(row['source'], row['target'])
        edge_label_dict[(row['source'], row['target'])] = weight_label
    
    # boolean plot
    if create_plot:
        # extract positions
        positions = nx.get_node_attributes(G, 'pos')
        
        # plot the graph
        nx.draw(G,
                pos=positions,
                with_labels=node_labels,
                node_size=node_size,
                node_color=node_color,
                arrows=True)
        
        # if edge labels desired
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_label_dict, font_size=8)
        
        # if save path given
        if save_path:
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
        
        # show the plot
        plt.show()
        
    return G

# sigmoid function
def sigmoid(z):
    '''
    Transforms a value into its Sigmoid function value;
    "squished" between 0 and 1.

    Parameters
    ----------
    z : integer or numpy array
        Number, vector, or matrix to transform.

    Returns
    -------
    g_z : integer or numpy array
        Transformed number, vector, or matrix.

    '''
    # calculate g(z)
    g_z = 1 / (1 + np.exp(-1 * z))
    
    # return g(z)
    return g_z

# function to create random weights and biases for the initial run
def create_random_parameters(architecture, random_state=None):
    '''
    Creates random weights and biases for the initial feed-forward loop
    of a neural network.

    Parameters
    ----------
    architecture : list of integers
        Number of nodes each layer of the neural network. Order matters.
        Example: architecture = [2, 3, 3, 1]
            - Input Layer: 2 nodes
            - Hidden Layer 1: 3 nodes
            - Hidden Layer 2: 3 nodes
            - Output Layer: 1 node
    random_state: integer
        Number to set random state seed for consistent results.

    Returns
    -------
    weights : dictionary
        Randomized weights for the layers in a neural network.
            - keys: weights_{layer} for each layer 1 to L
            - values: numpy array of randomized weights for layer 1 to L
            * dimensions: n_{l} x n_{l-1}; node count in l by node count in l-1
    biases : dictionary
        Randomized biases for the layers in a neural network.
            - keys: biases_{layer} for each layer 1 to L
            - values: numpy array of randomized biases for layer 1 to L
            * dimensions: n_{l} x 1; node count in l by 1

    '''
    
    # apply random state
    if random_state:
        # store current global random state
        global_random_state = np.random.get_state()
        
        # set seed
        np.random.seed(random_state)
    
    # random weights and biases - data structure for storage
    weights = {}
    biases = {}
    
    '''
    # completely random initialization
    # iterate backwards
    for layer in range(len(architecture) - 1, 0, -1):
        weights[f'weights_{layer}'] = np.random.randn(architecture[layer], architecture[layer - 1])
        biases[f'biases_{layer}'] = np.random.randn(architecture[layer], 1)
    '''
    
    # xavior weights initialization with zeros for bias initialization
    # iterate backwards
    for layer in range(len(architecture) - 1, 0, -1):
        # xavier initialization for weights (scaling helps prevent exploding/vanishing gradients)
        weights[f'weights_{layer}'] = np.random.randn(architecture[layer], architecture[layer - 1]) * np.sqrt(1 / architecture[layer - 1])
        
        # initalize biases to zero
        biases[f'biases_{layer}'] = np.zeros((architecture[layer], 1))
        
    # restore original random state if seed was set
    if random_state:
        np.random.set_state(global_random_state)
        
    return weights, biases

# function for feed forward process
def feed_forward(X, weights, biases):
    '''
    Models the feed forward process for a neural network.

    Parameters
    ----------
    X : numpy array
        Input values; raw; A0 transposes this for references
            * dimensions: m x n; rows by features; samples by features
            * input layer should have n nodes
    weights : dictionary
        Weights for the layers in a neural network.
            - keys: weights_{layer} for each layer 1 to L
            - values: numpy array of randomized weights for layer 1 to L
            * dimensions: n_{l} x n_{l-1}; node count in l by node count in l-1
            * first iteration should use the randomized values from create_random_parameters()
    biases : dictionary
        Biases for the layers in a neural network.
            - keys: biases_{layer} for each layer 1 to L
            - values: numpy array of randomized biases for layer 1 to L
            * dimensions: n_{l} x 1; node count in l by 1
            * first iteration should use the randomized values from create_random_parameters()

    Returns
    -------
    y_hat : numpy array
        Final output (prediction).
            * dimensions: n_{L} x m; node count in L by rows; node count in final layer by samples
    activated : dictionary
        - keys: activated_{layer} for each layer 1 to L
        - values: numpy array of values after activation function for each layer.
            * dimensions: n_{l} x m; node count in l by rows; node count in l by samples
            * A0 is not returned in this version

    '''

    # create parameter call functions (ensure proper order)
    weights_calls = [f'weights_{layer}' for layer in range(2, len(weights) + 1)]
    biases_calls = [f'biases_{layer}' for layer in range(2, len(biases) + 1)]
    
    # create data structure for activated values
    activated = {}
    
    # initialize first activated layer (transposed input)
    A = X.T
    
    # cache activated transposed values
    activated['activated_0'] = A
    
    # calculate Z
    Z = weights['weights_1'] @ A + biases['biases_1']
    
    # calculate g(Z) (activate with sigmoid)
    A = sigmoid(Z)
    
    # cache activated values
    activated['activated_1'] = A
    
    # iterate weights and biases
    for weights_l, biases_l in zip(weights_calls, biases_calls):
        # calculate Z
        Z = weights[weights_l] @ A + biases[biases_l]
        
        # calculate g(Z) (activate with sigmoid)
        A = sigmoid(Z)
        
        # cache activated values
        activated[weights_l.replace('weights', 'activated')] = A
    
    # final activate layer is y_hat (prediction)
    y_hat = A.copy()
    
    return y_hat, activated
    
# function to calculate cost with binary cross entropy (BCE)
def calculate_bce(y_hat, y):
    '''
    Calculate cost using binary cross entropy.

    Parameters
    ----------
    y_hat : numpy array
        Final output (prediction).
            * dimensions: n_{L} x m: node count in L by rows; node count in final layer by samples
    y : numpy array
        Observed values; actual values; training lables; technically Y.
            * dimensions: n_{L} x m: node count in L by rows; node count in final layer by samples

    Returns
    -------
    cost : float
        Cost of the neural network calculated using binary cross entropy.

    '''
    
    # losses with a small epsilon to prevent numerical instability (1e-15)
    losses = -((y * np.log(y_hat + 1e-15)) + (1 - y) * np.log(1 - y_hat + 1e-15))
    
    # get row dimension
    m = y_hat.reshape(-1).shape[0]
    
    # sum across rows
    summed_losses = (1 / m) * np.sum(losses, axis=1)
    
    # calculate cost
    cost = np.sum(summed_losses)
    
    # return
    return cost

# function for backpropagation
def backpropagation(y_hat, y, weights, biases, activated):
    '''
    Function to perform backpropagation

    Parameters
    ----------
    y_hat : numpy array
        Final output (prediction).
            * dimensions: n_{L} x m; node count in L by rows; node count in final layer by samples
    y : numpy array
        Observed values; actual values; training lables; technically Y.
            * dimensions: n_{L} x m: node count in L by rows; node count in final layer by samples
    weights : dictionary
        Weights for the layers in a neural network.
            - keys: weights_{layer} for each layer 1 to L
            - values: numpy array of randomized weights for layer 1 to L
            * dimensions: n_{l} x n_{l-1}; node count in l by node count in l-1
            * first iteration should use the randomized values from create_random_parameters()
    biases : dictionary
        Biases for the layers in a neural network.
            - keys: biases_{layer} for each layer 1 to L
            - values: numpy array of randomized biases for layer 1 to L
            * dimensions: n_{l} x 1; node count in l by 1
            * first iteration should use the randomized values from create_random_parameters()
    activated: dictionary
        - keys: activated_{layer} for each layer 1 to L
        - values: numpy array of values after activation function for each layer.
            * dimensions: n_{l} x m; node count in l by rows; node count in l by samples
            * A0 is not returned in this version

    Returns
    -------
    nabla_weights : dictionary
        Partial derivates of the cost function with respect to layer weights.
    nabla_biases : dictionary
        Partial derivates of the cost function with respect to layer biases.

    '''
    
    # create data structures for partial derivates of weights and biases
    nabla_weights = {}
    nabla_biases = {}
    propagators = {}
    
    # get layers information from activated
    layers_listed = list(activated.keys())
    layers_numeric = [int(layer_num.split('_')[1]) for layer_num in layers_listed]
    max_layer = max(layers_numeric)
    
    # get sample size / rows from y_hat
    m = y_hat.shape[1]
    
    # calculate weights and biases for final layer (L)
    # first component: dC/dZ_L = dC/dA_L * dA_L/dZ_L
    component_1 = (1 / m) * (activated[f'activated_{max_layer}'] - y)
    # second component: dZ_L/dW_L = A_{L-1} -> transpose for calculation
    component_2 = activated[f'activated_{max_layer - 1}']
    # partial weights: dC/dZ_L * A_{L-1}^T
    partial_weights = component_1 @ component_2.T
    # partial biases: dC/db_L = rowsums(dC/dZ_L)
    partial_biases = np.sum(component_1, axis=1, keepdims=True)
    # cache partials
    nabla_weights[f'partial_{max_layer}'] = partial_weights
    nabla_biases[f'partial_{max_layer}'] = partial_biases
    
    # calculate "propagator" for penultimate layer (L - 1)
    # propagator: dC/dA_{L-1} = dC/dZ_L * dZ_L/dA_{L-1}
    # dZ_L/dA_{L-1} = W_L -> transpose and swap for calculation
    propagator = weights[f'weights_{max_layer}'].T @ component_1
    propagators[f'propagator_{max_layer}'] = propagator
    
    # iterate backwards from l=L-1 to l=1, calculating weights and biases
    for layer in range(max_layer - 1, 0, -1):
        # propagation step: dC/dZ_l = dZ_{l+1}/dA_{l} * dA_l/dZ_l
        # first component: dA_l/dZ_l -> sigmoid derivative in terms of A_l
        # g(Z_l)' = g(Z_l) * (1 - g(Z_l)) = A_l * (1 - A_l)
        component_1 = activated[f'activated_{layer}'] * (1 - activated[f'activated_{layer}'])
        # second component: propagator_{l+1} = dZ_{l+1}/dA_{l}
        # propagates into the level dC/dZ_l to access weights and biases
        component_2 = propagator * component_1
        
        # calculate weights and biases for layer l
        # third component: dZ_l/dW_l = A_{l-1} -> transpose for calculation
        component_3 = activated[f'activated_{layer - 1}']
        # partial weights: dC/dZ_l * A_{l-1}^T
        partial_weights = component_2 @ component_3.T
        # partial biases: dC/db_l = rowsums(dC/dZ_l)
        # partial_biases = np.sum(partial_weights, axis=1, keepdims=True) # og code - mistake?
        partial_biases = np.sum(component_2, axis=1, keepdims=True)
        # cache partials
        nabla_weights[f'partial_{layer}'] = partial_weights
        nabla_biases[f'partial_{layer}'] = partial_biases
        
        # calculate "propagator" for next layer (l - 1)
        # propagator: dC/dA_{l-1} = dC/dZ_l * dZ_l/dA_{l-1}
        # dZ_l/dA_{l-1} = W_l -> transpose and swap for calculation
        propagator = weights[f'weights_{layer}'].T @ component_2
        propagators[f'propagator_{layer}'] = propagator
    
    return nabla_weights, nabla_biases

# function to raise errors for training input and architecture
def raise_architecture_errors(X, y, architecture, epochs, alpha, threshold, random_state):
    '''
    Checks data types, dimensions, and issues warnings for training the neural network.
    '''
    
    # check data types
    if not isinstance(X, np.ndarray):
        raise TypeError('X must be a numpy array.')
    if not isinstance(y, np.ndarray):
        raise TypeError('y must be a numpy array.')
    if not isinstance(architecture, list):
        raise TypeError('architecutre must be a list.')
    if not all(isinstance(layer, int) and layer > 0 for layer in architecture):
        raise ValueError('architecutre must be a list of positive integers.')
    if not isinstance(epochs, int):
        raise TypeError('epochs must be an integer.')
    if epochs <= 0:
        raise ValueError('epochs must be a postive integer.')
    if not isinstance(alpha, (float, int)):
        raise TypeError('alpha must be a float or an integer.')
    if alpha <= 0:
        raise ValueError('alpha must be a positive value.')
    if not isinstance(threshold, float):
        raise TypeError('threshold must be a float value.')
    if (threshold < 0) or (threshold > 1):
        raise ValueError('threshold must be a float value between 0 and 1.')
    if random_state is not None and not isinstance(random_state, int):
        raise TypeError('random_state must be an integer value (accepted values: positive, zero, or negative).')

    # check if input features match the architecture
    if X.shape[1] != architecture[0]:
        raise ValueError(f'Input features mismatch: X has {X.shape[1]} features, expected {architecture[0]}.')
    if architecture[-1] != 1:
        raise ValueError('Model currently only supports output layers of size 1.')
    if y.shape[0] != architecture[-1]:
        raise ValueError(f'Output features mismatch: y has {y.shape[0]} features, expected {architecture[-1]}.')
    if X.shape[0] != y.shape[1]:
        raise ValueError(f'Sample size mismatch: X has {X.shape[0]} samples, y has {y.shape[1]} samples.')
    
    # preemptive warnings
    if alpha > 1:
        print('Warning: High alpha values may cause unstable training.')

# function to train the neural network
def train(X, y, architecture, epochs=10, alpha=0.01, threshold=0.5, random_state=None):
    '''
    Function to train the neural network.

    Parameters
    ----------
    X : numpy array
        Input values; raw; A0 transposes this for references
            * dimensions: m x n; rows by features; samples by features
            * input layer should have n nodes
    y : numpy array
        Observed values; actual values; training lables; technically Y.
            * dimensions: n_{L} x m: node count in L by rows; node count in final layer by samples
    architecture : list of integers
        Number of nodes each layer of the neural network. Order matters.
        Example: architecture = [2, 3, 3, 1]
            - Input Layer: 2 nodes
            - Hidden Layer 1: 3 nodes
            - Hidden Layer 2: 3 nodes
            - Output Layer: 1 node
    epochs : integer, optional
        The number of iterations to update the model parameters. The default is 10.
    alpha : float, optional
        Learning rate (step-size) to update the model parameters. The default is 0.01.
    threshold : float, optional
        Threshold for binary classifcation. The default is 0.5.
    random_state: integer
        Number to set random state seed for consistent results.

    Raises
    ------
    architecture
        Checks data types, dimensions, and issues warnings.

    Returns
    -------
    weights : dictionary
        Weights for the layers in a neural network.
            - keys: weights_{layer} for each layer 1 to L
            - values: numpy array of randomized weights for layer 1 to L
            * dimensions: n_{l} x n_{l-1}; node count in l by node count in l-1
    biases : dictionary
        Biases for the layers in a neural network.
            - keys: biases_{layer} for each layer 1 to L
            - values: numpy array of randomized biases for layer 1 to L
            * dimensions: n_{l} x 1; node count in l by 1
    metrics : dictionary
        - costs: binary cross entropy over the epochs (iterations).
        - accuracies: accuracy on the training set over the epochs.
        - splits: time splits over the epochs.
        - cost: final cost.
        - accuracy: final accuracy.
        - elapsed: total time to train.
        
    '''
    
    # raise architecture errors
    raise_architecture_errors(X, y, architecture, epochs, alpha, threshold, random_state)
    
    # create data structure for metrics
    costs = []
    accuracies = []
    splits = []
    
    # create random parameters
    weights, biases = create_random_parameters(architecture, random_state)
    
    # get layers to update
    update_layers = len(architecture)
    
    # time tracker start
    start_time = time.time()
    
    # iterate through the epochs
    for epoch in range(epochs):
        # split time
        split_start = time.time()
        
        # feed forward
        y_hat, activated = feed_forward(X, weights, biases)
        
        # calculate cost
        cost = calculate_bce(y_hat, y)
        
        # track cost
        costs.append(cost)
        
        # calculate accuracy
        accuracy = np.mean((y_hat > threshold) == y) * 100
        
        # track accuracy
        accuracies.append(accuracy)
        
        # backpropagation
        nabla_weights, nabla_biases= backpropagation(y_hat, y, weights, biases, activated)
        
        # update parameters
        for layer in range(1, update_layers):
            # update weights
            weights[f'weights_{layer}'] -= alpha * nabla_weights[f'partial_{layer}']
            
            # update biases
            biases[f'biases_{layer}'] -= alpha * nabla_biases[f'partial_{layer}']
        
        # split time
        split_end = time.time() - split_start
        
        # track splits
        splits.append(split_end)
    
    # time tracker end
    elapsed = time.time() - start_time
    
    # report final metrics
    print(f'Final Cost: {cost}\nFinal Accuracy: {accuracy}\nElapsed: {elapsed}')
    
    # aggregate final metrics
    metrics = {'costs': costs,
               'accuracies': accuracies,
               'splits': splits,
               'cost': cost,
               'accuracy': accuracy,
               'elapsed': elapsed}
            
    return weights, biases, metrics

# function to predict values
def predict(X, weights, biases, threshold=0.5, flatten_output=True):
    '''
    Uses the fitted parameters from the neural network to make predictions.

    Parameters
    ----------
    X : numpy array
        Input values; raw features.
            * dimensions: m x n; rows by features; samples by features
    weights : dictionary
        Trained weights for the layers in the neural network.
    biases : dictionary
        Trained biases for the layers in the neural network.
    threshold : float, optional
        Threshold for binary classifcation. The default is 0.5.
    flatten_output: boolean, optional
        If True, returns predictions as a 1D array. The default is True.

    Returns
    -------
    predictions : numpy array
        Binary predictions from the output layer.
    raw_predictions : numpy array
        Raw predictions from the output layer.
    
    '''
    
    # feed forward for probabilities
    raw_predictions, _ = feed_forward(X, weights, biases)
    
    # convert probabilities to binary predictions
    predictions = (raw_predictions > threshold).astype(int)
    
    # flatten predictions
    if flatten_output:
        raw_predictions = raw_predictions.ravel()
        predictions = predictions.ravel()
    
    # return probabilities and binary predictions
    return predictions, raw_predictions
