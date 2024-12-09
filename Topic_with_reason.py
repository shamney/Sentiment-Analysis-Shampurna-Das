import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from matplotlib.dates import DateFormatter

# Load the dataset
file_path = "~/Desktop/processed_reviews.csv"
df = pd.read_csv(file_path, usecols=["asin", "Datetime", "processed_text"])  # Load only necessary columns

# Ensure the data is sorted by datetime
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
df = df.dropna(subset=['Datetime'])
df.sort_values(by='Datetime', inplace=True)

# Filter data for specific ASINs early to reduce dataset size
asins_to_analyze = ["B07577WHPQ", "B005IYYF5E", "B083BDVS36"]
df = df[df['asin'].isin(asins_to_analyze)]

# Calculate sentiment from the processed_text column
df['sentiment'] = df['processed_text'].apply(lambda text: TextBlob(str(text)).sentiment.polarity)

# Ensure the processed_text column is a string for BERTopic
df['processed_text'] = df['processed_text'].astype(str)
documents = df['processed_text'].tolist()

# Initialize and fit BERTopic with optimized parameters
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Smaller, faster embedding model
topic_model = BERTopic(
    embedding_model=embedding_model,
    verbose=True,
    nr_topics=15,  # Adjusted for performance
    min_topic_size=30,  # Larger minimum topic size for faster processing
    top_n_words=5,  # Focus on fewer words per topic for simplicity
    calculate_probabilities=True
)

# Fit the topic model
topics, probs = topic_model.fit_transform(documents)

# Add topics to the dataframe
df['topic'] = topics

# Aggregate sentiment and topics at weekly intervals
weekly_sentiment_topics = df.groupby(['asin', pd.Grouper(key='Datetime', freq='W')]) \
    .agg({'sentiment': 'mean', 'topic': lambda x: list(x)}).reset_index()

# Explode the topics into individual rows for frequency calculation
weekly_sentiment_topics_exploded = weekly_sentiment_topics.explode('topic')

# Generate dynamic topic names based on the top words for each topic
top_words = topic_model.get_topics()
topic_names = {topic_num: " ".join([word[0] for word in words[:5]]) for topic_num, words in top_words.items()}

# Function to analyze and plot data for each ASIN
def analyze_asin(asin):
    asin_data = weekly_sentiment_topics_exploded[weekly_sentiment_topics_exploded['asin'] == asin]
    if asin_data.empty:
        print(f"No data found for ASIN: {asin}")
        return

    # Calculate average sentiment over time
    sentiment_trend = asin_data.groupby('Datetime')['sentiment'].mean()

    # Calculate topic frequency over time
    topic_frequency = asin_data.groupby(['Datetime', 'topic']).size().unstack(fill_value=0)

    # Select the top 5 topics based on overall frequency
    top_topics = topic_frequency.sum().sort_values(ascending=False).head(5).index
    topic_frequency = topic_frequency[top_topics]

    # Plot sentiment and topic trends
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot sentiment trend
    ax1.plot(sentiment_trend.index, sentiment_trend, color='blue', label='Average Sentiment', marker='o')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Average Sentiment', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(f"Sentiment Trends and Topic Frequencies for ASIN {asin}")

    # Add a secondary axis for topic frequencies
    ax2 = ax1.twinx()
    for topic in topic_frequency.columns:
        topic_name = topic_names.get(topic, f'Topic {topic}')
        ax2.plot(topic_frequency.index, topic_frequency[topic], label=f"{topic_name}")

    ax2.set_ylabel('Topic Frequency', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Format the x-axis as dates
    date_format = DateFormatter("%Y-%m")
    ax1.xaxis.set_major_formatter(date_format)

    # Add a legend for both axes
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
    fig.tight_layout()

    # Show plot
    plt.grid()
    plt.show()

# Analyze each ASIN
for asin in asins_to_analyze:
    analyze_asin(asin)

print("Analysis complete.")
