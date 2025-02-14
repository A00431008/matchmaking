import csv
import umap
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from collections import defaultdict
import pyvis
from pyvis.network import Network
import numpy as np
import seaborn as sns
import branca.colormap as cm
import branca
import pandas as pd
import re
from textwrap import wrap
import json
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

project_path = "./"

with open(project_path + "umap_tuning_results.json", "r") as f:
    tuning_results = json.load(f)

best_params = tuning_results["best_params"]  # Extracting the best parameters from JSON file

n_neighbors = best_params["n_neighbors"]
spread = best_params["spread"]
min_dist = best_params["min_dist"]

print(f"Using Best UMAP Parameters: n_neighbors={n_neighbors}, spread={spread:.4f}, min_dist={min_dist:.4f}")

# Read attendees and their responses from a CSV file, replace attendees.csv with own link or file name
attendees_map = {}
with open(project_path + 'classmates.csv', newline='') as csvfile:
    attendees = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(attendees)  # Skip the header row
    for row in attendees:
        name, paragraph = row
        attendees_map[paragraph] = name

# Generate sentence embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
paragraphs = list(attendees_map.keys())
embeddings = model.encode(paragraphs)

# Create a dictionary to store embeddings for each person
person_embeddings = {attendees_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}

# Inputting the tuned UMAP parameters
reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    spread=spread,
    metric="cosine",
    random_state=30)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(list(person_embeddings.values()))
reduced_data = reducer.fit_transform(scaled_data)

# Creating lists of coordinates with accompanying labels
x = [row[0] for row in reduced_data]
y = [row[1] for row in reduced_data]
label = list(person_embeddings.keys())

# Plotting and annotating data points
plt.scatter(x,y)
for i, name in enumerate(label):
    plt.annotate(name, (x[i], y[i]), fontsize="3")

# Clean-up and Export
plt.axis('off')
plt.savefig(project_path+'visualization_tuned_30.png', dpi=800)

# Computing cosine similarity
cosine_sim_matrix = cosine_similarity(scaled_data)

# Computing the Euclidean distance in the 2D UMAP space
euclidean_dist_matrix = squareform(pdist(reduced_data, metric="euclidean"))

# Computing the Spearman Rank Correlation between cosine similarity and Euclidean distance
correlations = []
for i in range(len(scaled_data)):
    cosine_ranks = np.argsort(-cosine_sim_matrix[i]) 
    euclidean_ranks = np.argsort(euclidean_dist_matrix[i]) 
    spearman_corr, _ = spearmanr(cosine_ranks, euclidean_ranks)

    if not np.isnan(spearman_corr): 
        correlations.append(spearman_corr)

# Calculating average Spearman correlation across all students
avg_spearman = np.mean(correlations) if correlations else 0

print(f"\nSpearman Rank Correlation (ρ) between cosine similarity and Euclidean distance: {avg_spearman:.4f}")