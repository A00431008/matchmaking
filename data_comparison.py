
import csv
from sentence_transformers import SentenceTransformer
import numpy as np
from textwrap import wrap
from sklearn.metrics.pairwise import cosine_similarity

project_path = "./"

# Read attendees and their responses from a CSV file, replace attendees.csv with own link or file name
attendees_map = {}
with open(project_path + 'classmates_modified.csv', newline='') as csvfile:
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
np.save("modified_classmates_embeddings.npy", person_embeddings)


#============================COMPARISION================================================================================

# Load old and new embeddings
original_embeddings = np.load("classmates_embeddings.npy", allow_pickle=True).item()
modified_embeddings = np.load("modified_classmates_embeddings.npy", allow_pickle=True).item()

# Define names of modified individuals
modified_people = ["Sudeep Raj Badal", "Mohammed Abdul Thoufiq", "Louise Fear"]

# Compare embeddings for modified people
for person in modified_people:
    if person in original_embeddings and person in modified_embeddings:
        original_vector = original_embeddings[person].reshape(1, -1)
        modified_vector = modified_embeddings[person].reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(original_vector, modified_vector)[0][0]
        print(f"Similarity for {person}: {similarity:.4f}")