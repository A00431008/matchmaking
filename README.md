# Embedding Matchmaking

_"Words can't describe how unique your interests are... but coordinates can" - Sean Ashley, circa 2023_

A flattened embedding space of names clustered based on their interests using the sentence-transformers all-MiniLM-L6-v2 model. Created for the UW Startups S23 Kickoff event with guidance from [Jacky Zhao](https://jzhao.xyz/) and [Sean Ashley](https://www.linkedin.com/in/sean-ashley). [Simha Kalimipalli](https://github.com/Simha-Kalimipalli) later aded interactivity!

![Sample output of script](https://github.com/A00431008/matchmaking/blob/main/visualization.png)

## Instructions for use

1. Collect or format your data in the following format

| Name  | What are your interests? (or varying permutations of this question) |
| ----- | ------------------------------------------------------------------- |
| Alice | I love being the universal placeholder for every CS joke ever       |
| Bob   | I too love being the universal placeholder for every CS joke        |

2. Clone the repository
3. Install all required packages using pip or conda:

- `umap-learn`
- `scikit-learn`
- `scipy`
- `sentence-transformers`
- `matplotlib`
- `pyvis`
- `pandas`
- `numpy`
- `seaborn`
- `branca`

4. Replace `attendees.csv` in `visualizer.ipynb` with the path to your downloaded data
5. Run all cells
6. Bask in the glory of having an awesome new poster
7. Make two (!) cool interactive visualizations

## Data Analysis

To analyze the impact of sentence modifications on embeddings, we selected three sentences and made both **minor** (synonym replacement) and **major** (sentence restructuring) changes. We then computed the cosine similarity between the original and modified embeddings to assess how much the meaning was affected.

### Changes Made:
| Name                     | Original Description | Modified Description |
|--------------------------|----------------------|----------------------|
| **Sudeep Raj Badal**  | "I like listening to music, reading, going on long drives and watching movies." | "Exploring new places on long drives, immersing myself in books, and enjoying music and films are my favorite pastimes." |
| **Mohammed Abdul Thoufiq**  | "I like watching movies, playing cricket, efootball and collecting sneakers." | "I have a passion for cricket and gaming, enjoy collecting sneakers, and love immersing myself in great films." |
| **Louise Fear**  | "I like reading, playing video games, and baking." | "Baking, reading books, and playing video games are my favorite hobbies." |

### Results:
| Name                     | Cosine Similarity |
|--------------------------|------------------|
| **Sudeep Raj Badal**     | **0.7096** (Significant change) |
| **Mohammed Abdul Thoufiq** | **0.7868** (Moderate change) |
| **Louise Fear**          | **0.8755** (Minimal change) |

### Insights:
1. **Minor wording changes (synonyms, slight rephrasing) resulted in high similarity (~0.87).**  
   - Example: "playing video games" → "enjoying video games" had minimal impact.  

2. **Sentence restructuring had a moderate effect (~0.78).**  
   - Example: "watching movies, playing cricket, efootball, and collecting sneakers" → "passion for cricket and gaming, enjoy collecting sneakers, and love immersing myself in great films" introduced new structure but kept core ideas.  

3. **Conceptual shifts caused the largest changes (~0.70).**  
   - Example: "going on long drives" → "exploring new places on long drives" changed the meaning enough to significantly alter the embedding representation.  


The results indicate that while embeddings are resistant to minor rewording, **structural modifications and conceptual changes significantly impact similarity scores**. This is crucial for applications like clustering, retrieval, and text classification, where sentence meaning must be preserved.


## Embedding Sensitivity Tests
We tested two sentence embedding models, all-MiniLM-L6-v2 and all-mpnet-base-v2, to see how differently they measure similarity based on personal descriptions. While both models give similar overall rankings, they sometimes change the order of who is most similar.

### My Description:
"I like watching movies, playing cricket, eFootball, and collecting sneakers."

### Quantitative Analysis:
The two models mostly agree, but there are differences. For example, Ethan Cooke was ranked 3rd by Model 1 but dropped to 8th in Model 2. Meanwhile, Somto Muotoe moved up from 11th to 3rd, and Sriram Ramesh improved from 8th to 4th. This suggests that each model picks up on different aspects of similarity.

### Qualitative Observations:
Model 2 (all-mpnet-base-v2) seems to understand deeper connections, changing who is considered more similar. For example, Somto Muotoe (who likes reading, cycling, and video games) was ranked low in Model 1 but much higher in Model 2, likely because Model 2 connects multiple interests better. On the other hand, Ethan Cooke (who enjoys hiking and board games) dropped in ranking, suggesting that Model 1 (MiniLM) may have focused more on individual words rather than overall meaning.

These differences show that choosing the right model matters. While both models give similar results, small ranking changes can affect how we group people based on their interests.
