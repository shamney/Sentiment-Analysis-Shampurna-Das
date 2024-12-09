import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
file_path = "~/Desktop/processed_reviews.csv"
df = pd.read_csv(file_path)

# Ensure the data is sorted by the existing datetime column
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')  # Convert to datetime, handle errors gracefully
df = df.dropna(subset=['Datetime'])  # Drop rows with invalid datetime
df.sort_values(by='Datetime', inplace=True)

# Prepare data for Doc2Vec
tagged_data = [TaggedDocument(words=str(text).split(), tags=[asin]) for text, asin in zip(df['processed_text'], df['asin'])]

# Train a Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# Generate sentiment scores using cosine similarity with positive/negative reference
positive_vector = doc2vec_model.infer_vector(["good", "excellent", "satisfied", "happy","wow", "positive"])
negative_vector = doc2vec_model.infer_vector(["bad", "poor", "unsatisfied", "angry", "negative"])

def compute_sentiment(doc):
    vector = doc2vec_model.infer_vector(str(doc).split())
    similarity_pos = cosine_similarity([vector], [positive_vector])[0, 0]
    similarity_neg = cosine_similarity([vector], [negative_vector])[0, 0]
    return similarity_pos - similarity_neg  # Positive sentiment > 0, Negative sentiment < 0

# Apply Doc2Vec sentiment computation to the dataset
df['sentiment'] = df['processed_text'].apply(compute_sentiment)

# Aggregate sentiment function
def aggregate_sentiment(df, freq):
    return df.groupby(['asin', pd.Grouper(key='Datetime', freq=freq)])['sentiment'].mean().reset_index()

# Aggregate daily, weekly, monthly, and yearly sentiment
daily_sentiment = aggregate_sentiment(df, 'D')
weekly_sentiment = aggregate_sentiment(df, 'W')
monthly_sentiment = aggregate_sentiment(df, 'M')
yearly_sentiment = aggregate_sentiment(df, 'Y')

# List of ASINs to analyze
asins_to_analyze = ["B07577WHPQ", "B005IYYF5E", "B083BDVS36"]

# Loop through each ASIN in the list
for asin_to_plot in asins_to_analyze:
    if asin_to_plot in df['asin'].unique():
        print(f"Analyzing sentiment trends for ASIN: {asin_to_plot}")

        # Helper function to plot sentiment trends
        def plot_sentiment(data, freq, title):
            if not data.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(data['Datetime'], data['sentiment'], marker='o', label=f"{freq} Sentiment")
                plt.title(f"{title} Sentiment Trends for ASIN {asin_to_plot}")
                plt.xlabel("Date")
                plt.ylabel("Average Sentiment")
                plt.legend()
                plt.grid()
                plt.show()
            else:
                print(f"No {freq.lower()} sentiment data available for ASIN {asin_to_plot}")

        # Visualize daily, weekly, monthly, and yearly sentiment
        asin_daily = daily_sentiment[daily_sentiment['asin'] == asin_to_plot]
        plot_sentiment(asin_daily, "Daily", "Daily")

        asin_weekly = weekly_sentiment[weekly_sentiment['asin'] == asin_to_plot]
        plot_sentiment(asin_weekly, "Weekly", "Weekly")

        asin_monthly = monthly_sentiment[monthly_sentiment['asin'] == asin_to_plot]
        plot_sentiment(asin_monthly, "Monthly", "Monthly")

        asin_yearly = yearly_sentiment[yearly_sentiment['asin'] == asin_to_plot]
        plot_sentiment(asin_yearly, "Yearly", "Yearly")

        # Time-series forecasting with LSTM for Weekly Sentiment
        if not asin_weekly.empty:
            # Normalize sentiment values
            scaler = MinMaxScaler(feature_range=(0, 1))
            sentiment_scaled = scaler.fit_transform(asin_weekly[['sentiment']])

            # Prepare data for LSTM
            def create_lstm_data(series, time_steps):
                X, y = [], []
                for i in range(len(series) - time_steps):
                    X.append(series[i:i + time_steps, 0])  # Create sequences of time_steps length
                    y.append(series[i + time_steps, 0])  # Target is the next value
                return np.array(X), np.array(y)

            # Number of time steps for LSTM
            time_steps = 5
            X, y = create_lstm_data(sentiment_scaled, time_steps)

            if X.shape[0] > 0 and y.shape[0] > 0:
                # Reshape for LSTM input (samples, time_steps, features)
                X = X.reshape((X.shape[0], X.shape[1], 1))

                # Split into training and testing data
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                # Build LSTM model
                model = Sequential([
                    LSTM(50, activation='relu', input_shape=(time_steps, 1)),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')

                # Train LSTM model
                model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

                # Predict sentiment
                y_pred = model.predict(X_test)

                # Inverse scale predictions and actual values
                y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
                y_pred_rescaled = scaler.inverse_transform(y_pred)

                # Plot actual vs predicted sentiment
                plt.figure(figsize=(12, 6))
                plt.plot(range(len(y_test_rescaled)), y_test_rescaled, label='Actual Sentiment', marker='o')
                plt.plot(range(len(y_pred_rescaled)), y_pred_rescaled, label='Predicted Sentiment', marker='x')
                plt.title(f"Sentiment Prediction for ASIN {asin_to_plot} using LSTM")
                plt.xlabel("Time Steps")
                plt.ylabel("Sentiment")
                plt.legend()
                plt.grid()
                plt.show()

            else:
                print("Not enough data for LSTM model.")
        else:
            print(f"No weekly sentiment data available for ASIN {asin_to_plot}")
    else:
        print(f"ASIN {asin_to_plot} not found in dataset.")
