import csv
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

# Initialize project path and classmate file path
project_path = "./"

# Read the CSV containing classmates and their descriptions
attendees_map = {}
with open(project_path + 'classmates.csv', newline='') as csvfile:
    attendees = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(attendees)  # Skip the header row
    for row in attendees:
        name, paragraph = row
        attendees_map[paragraph] = name

# Initialize the models
model1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model2 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Your description
my_description = "I like watching movies, playing cricket, efootball and collecting sneakers"  # Your description

# Create embeddings for your own description and for all classmates
paragraphs = list(attendees_map.keys())
embeddings1 = model1.encode(paragraphs + [my_description])  # Add your description to embeddings
embeddings2 = model2.encode(paragraphs + [my_description])

# Split embeddings for classmates and yourself
classmate_embeddings1 = embeddings1[:-1]
my_embedding1 = embeddings1[-1]

classmate_embeddings2 = embeddings2[:-1]
my_embedding2 = embeddings2[-1]

# Compute similarity (cosine similarity) between you and each classmate for both models
similarities1 = [1 - cosine(my_embedding1, embedding) for embedding in classmate_embeddings1]
similarities2 = [1 - cosine(my_embedding2, embedding) for embedding in classmate_embeddings2]

# Rank the classmates based on similarity (from closest to farthest)
ranked_classmates1 = np.argsort(similarities1)[::-1]  # Sort in descending order (closest to farthest)
ranked_classmates2 = np.argsort(similarities2)[::-1]  # Sort in descending order (closest to farthest)

# Compute Spearman's rank correlation between the two models' rankings
spearman_corr, _ = spearmanr(ranked_classmates1, ranked_classmates2)

# Prepare the results for saving to CSV
similarity_data = []
for i, name in enumerate(attendees_map.values()):
    similarity_data.append({
        'Name': name,
        'Description': list(attendees_map.keys())[i],
        'Similarity_Model_1': similarities1[i],
        'Similarity_Model_2': similarities2[i],
        'Rank_Model_1': np.where(ranked_classmates1 == i)[0][0] + 1,  # Rank starts from 1
        'Rank_Model_2': np.where(ranked_classmates2 == i)[0][0] + 1,  # Rank starts from 1
    })

# Convert the results to a DataFrame
import pandas as pd
df = pd.DataFrame(similarity_data)

# Save the results to a CSV file
df.to_csv("model_comparison_results.csv", index=False)

# Print Spearman's Rank Correlation
print(f"Spearman's Rank Correlation: {spearman_corr:.4f}")
print("Results saved to model_comparison_results.csv")

# Optional: Print the ranked classmates for both models
print("\nRanked classmates based on Model 1 ('all-MiniLM-L6-v2'):")
for idx in ranked_classmates1:
    print(f"{attendees_map[list(attendees_map.keys())[idx]]}: {similarities1[idx]:.4f}")

print("\nRanked classmates based on Model 2 ('all-mpnet-base-v2'):")
for idx in ranked_classmates2:
    print(f"{attendees_map[list(attendees_map.keys())[idx]]}: {similarities2[idx]:.4f}")
print(df)