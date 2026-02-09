# !pip install scikit-network

import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sknetwork.clustering import Louvain, Leiden
from sklearn.preprocessing import MinMaxScaler

def run_graph_clustering(X_data, n_neighbors=30, method='leiden'):
    """
    Treats customers as a Network Graph and finds Communities.
    
    1. Builds a sparse KNN Graph (Who is similar to whom?)
    2. Optimizes Modularity (finding tight cliques)
    
    Methods: 'louvain' (Standard) or 'leiden' (Faster/Better convergence)
    """
    print(f"--- Starting Graph Clustering ({method.upper()}) ---")
    
    # 1. Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # 2. Build the Graph (Sparse Matrix)
    # Connect every user to their 30 nearest neighbors
    print(f"Building KNN Graph (k={n_neighbors})...")
    adjacency_matrix = kneighbors_graph(
        X_scaled, 
        n_neighbors=n_neighbors, 
        mode='connectivity', 
        include_self=False, 
        n_jobs=-1
    )
    
    # 3. Run Community Detection
    print(f"Detecting Communities using {method.upper()}...")
    if method == 'louvain':
        algo = Louvain(random_state=42)
    elif method == 'leiden':
        algo = Leiden(random_state=42)
        
    labels = algo.fit_predict(adjacency_matrix)
    
    # Re-index labels to be 0,1,2... sorted by size
    # (Graph algos often give random label IDs)
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_map = {lbl: new_id for new_id, lbl in enumerate(unique_labels[np.argsort(-counts)])}
    final_labels = np.array([sorted_map[l] for l in labels])
    
    print(f"Found {len(unique_labels)} Communities.")
    return final_labels

# Usage:
# labels_graph = run_graph_clustering(X_clean, method='leiden')  
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

def run_autoencoder_clustering(X_data, n_clusters=5, encoding_dim=10):
    print(f"--- Training Autoencoder on {len(X_data)} rows ---")
    
    # 1. Scale Data (Neural Nets need 0-1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # 2. Build the Autoencoder Network
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    
    # Encoder: Compresses data
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    bottleneck = Dense(encoding_dim, activation='relu')(encoded) # The "Latent Space"
    
    # Decoder: Reconstructs data (to learn the patterns)
    decoded = Dense(32, activation='relu')(bottleneck)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, output_layer)
    encoder = Model(input_layer, bottleneck) # We only need this half later
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # 3. Train (Mini-batch makes it memory safe for 1.8M rows)
    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=10,
        batch_size=4096,
        shuffle=True,
        verbose=1
    )
    
    # 4. Extract the "Latent Features"
    print("Extracting Deep Features...")
    X_encoded = encoder.predict(X_scaled, batch_size=4096)
    
    # 5. Cluster the Latent Space
    print("Clustering Latent Space...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=42)
    labels = kmeans.fit_predict(X_encoded)
    
    return labels, X_encoded

# Usage:
# labels_ae, latent_features = run_autoencoder_clustering(X_clean, n_clusters=5) 

from sklearn.neighbors import kneighbors_graph
from sknetwork.clustering import Louvain # pip install scikit-network
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp

def run_louvain_clustering(X_data, n_neighbors=15):
    print(f"--- Building Graph for Louvain (N={len(X_data)}) ---")
    
    # 1. Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # 2. Build KNN Graph (Sparse Matrix)
    # This connects every user to their 15 most similar users
    adjacency = kneighbors_graph(
        X_scaled, 
        n_neighbors=n_neighbors, 
        mode='connectivity', 
        include_self=False, 
        n_jobs=-1
    )
    
    # 3. Run Louvain
    # resolution: Controls size of clusters (1.0 = standard, >1 = smaller clusters)
    print("Running Louvain Community Detection...")
    louvain = Louvain(resolution=1.0)
    labels = louvain.fit_transform(adjacency)
    
    n_found = len(set(labels))
    print(f"Louvain found {n_found} natural communities.")
    
    return labels

# Usage:
# labels_louvain = run_louvain_clustering(X_clean)


from minisom import MiniSom # pip install minisom
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def run_som_clustering(X_data, grid_size=20):
    print(f"--- Training Self-Organizing Map (SOM) ---")
    
    # 1. Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # 2. Initialize SOM (e.g., 20x20 grid)
    som = MiniSom(
        x=grid_size, y=grid_size, 
        input_len=X_scaled.shape[1], 
        sigma=1.0, learning_rate=0.5
    )
    som.random_weights_init(X_scaled)
    
    # 3. Train (Iterative)
    print("Training SOM (This learns the topology)...")
    som.train_random(X_scaled, num_iteration=len(X_data)) # 1 pass through data
    
    # 4. Extract Clusters
    # Each node (0,0) to (20,20) is a micro-cluster.
    # We assign each user to their "Best Matching Unit" (BMU)
    print("Mapping users to SOM grid...")
    winner_coordinates = np.array([som.winner(x) for x in X_scaled])
    
    # Convert (x,y) coordinates to a single Cluster ID integer
    labels = np.ravel_multi_index(winner_coordinates.T, (grid_size, grid_size))
    
    return labels

# Usage:
# labels_som = run_som_clustering(X_clean)
