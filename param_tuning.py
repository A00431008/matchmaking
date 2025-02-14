import numpy as np
import umap
import optuna
import json
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import csv
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

project_path = "./"

# Loading Data
attendees_map = {}
with open(project_path + 'classmates.csv', newline='') as csvfile:
    attendees = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(attendees) 
    for row in attendees:
        name, paragraph = row
        attendees_map[paragraph] = name

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
paragraphs = list(attendees_map.keys())
embeddings = model.encode(paragraphs)

person_embeddings = {attendees_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}

scaler = StandardScaler()
scaled_data = scaler.fit_transform(list(person_embeddings.values()))

# Computing cosine similarity
cosine_sim_matrix = cosine_similarity(scaled_data)

# Store all trial results
trial_results = []

def objective(trial):
    """Optimize UMAP parameters using Spearman correlation."""

    # Controlled parameter search space
    n_neighbors = trial.suggest_int("n_neighbors", 6, 12)  # Restrict range
    spread = trial.suggest_float("spread", 1.85, 2.0)  # Very narrow range
    min_dist = trial.suggest_float("min_dist", 0.3, 0.5)  # Small controlled range

    umap_reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, spread=spread, metric="cosine",
        random_state=42  
    )

    low_dim_embeddings = umap_reducer.fit_transform(scaled_data)
    euclidean_dist_matrix = squareform(pdist(low_dim_embeddings, metric="euclidean"))

    correlations_spearman = []

    # Compute Spearman Rank Correlation for Each Student
    for i in range(len(scaled_data)):
        cosine_ranks = np.argsort(-cosine_sim_matrix[i])  
        euclidean_ranks = np.argsort(euclidean_dist_matrix[i]) 

        spearman_corr, _ = spearmanr(cosine_ranks, euclidean_ranks)

        if not np.isnan(spearman_corr): 
            correlations_spearman.append(spearman_corr)

    avg_spearman = np.mean(correlations_spearman) if correlations_spearman else 0

    # Storing results for logging
    trial_results.append({
        "trial": trial.number,
        "n_neighbors": n_neighbors,
        "spread": spread,
        "min_dist": min_dist,
        "spearman_corr": avg_spearman
    })


    print(f"Trial {trial.number}: n_neighbors={n_neighbors}, spread={spread:.4f}, min_dist={min_dist:.4f} → Spearman ρ: {avg_spearman:.4f}")

    return avg_spearman  

# Run Optuna optimization
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=200)  # Run 200 trials

# Get the best parameters
best_params = study.best_params
best_params["spearman_corr"] = study.best_value 

# Saving the best parameters and trial results to JSON
results_data = {
    "trials": trial_results,
    "best_params": best_params
}
with open(project_path + "umap_tuning_results.json", "w") as f:
    json.dump(results_data, f, indent=4)

print("\n=== Optimization Complete ===")
print(f"Final Best UMAP Parameters: {best_params}")
print("All trial results saved to 'umap_tuning_results.json'.")