import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import os

# Step 1: Load preprocessed data
file_path = os.path.expanduser("~/Desktop/processed_reviews.csv")
reviews = pd.read_csv(file_path)

# Ensure the file contains 'asin' and 'processed_text' columns
if 'asin' not in reviews.columns or 'processed_text' not in reviews.columns:
    raise ValueError("The file must contain 'asin' and 'processed_text' columns.")

# Convert 'processed_text' to lists of words if necessary
# Assuming the column is a string of space-separated words, convert to list
reviews['processed_text'] = reviews['processed_text'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Step 2: Remove rows with empty 'processed_text'
reviews = reviews[reviews['processed_text'].apply(lambda x: len(x) > 0)]

# Step 3: Tag data for Doc2Vec
tagged_data = [TaggedDocument(words=row['processed_text'], tags=[str(row['asin'])]) for _, row in reviews.iterrows()]

# Step 4: Train Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# Step 5: Infer vectors for each review
reviews['Review_Vector'] = reviews['processed_text'].apply(lambda x: doc2vec_model.infer_vector(x))

# Display the first few feature vectors
print("Extracted Features (Doc2Vec Vectors) for the First Few Reviews:")
print(reviews[['asin', 'Review_Vector']].head())

# Assign sentiment polarity manually or via another method (Example: Placeholder for positive, negative, neutral)
# For demonstration, let's assign random sentiments:
np.random.seed(42)
reviews['sentiment'] = np.random.choice(['positive', 'negative', 'neutral'], size=len(reviews))

# Define a custom color palette
sentiment_colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}

# Step 6: PCA Visualization
# Apply PCA to reduce vectors to 2D for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(reviews['Review_Vector'].tolist())

# Add PCA components to the DataFrame
reviews['PCA1'] = pca_components[:, 0]
reviews['PCA2'] = pca_components[:, 1]

# Plot PCA Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PCA1',
    y='PCA2',
    data=reviews,
    hue='sentiment',
    palette=sentiment_colors
)
plt.title('2D PCA Visualization of Review Vectors (Doc2Vec)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title="Sentiment")
plt.show()

# Step 7: Optional t-SNE Visualization
# Convert list of vectors to NumPy array
review_vectors_array = np.array(reviews['Review_Vector'].tolist())

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_components = tsne.fit_transform(review_vectors_array)

# Add t-SNE components to the DataFrame
reviews['TSNE1'] = tsne_components[:, 0]
reviews['TSNE2'] = tsne_components[:, 1]

# Plot t-SNE Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='TSNE1',
    y='TSNE2',
    data=reviews,
    hue='sentiment',
    palette=sentiment_colors
)
plt.title('2D t-SNE Visualization of Review Vectors (Doc2Vec)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title="Sentiment")
plt.show()
