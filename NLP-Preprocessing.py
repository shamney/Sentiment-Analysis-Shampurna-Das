import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure necessary resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function for text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""  # Handle non-string values by returning an empty string

    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join words back into a single string
    return ' '.join(words)

# File paths
input_file_path = '~/Desktop/updatedreviews.csv'
output_file_path = '~/Desktop/processed_reviews.csv'

# Read the CSV file
df = pd.read_csv(input_file_path)

# Apply preprocessing to the text column
df['processed_text'] = df['text'].apply(preprocess_text)

# Check for missing values in 'asin', 'datetime', and 'processed_text' columns
columns_to_check = ['asin', 'Datetime', 'processed_text']
null_count = df[columns_to_check].isnull().sum()
print(f"Null values in the specified columns:\n{null_count}")

# Drop rows with null values in 'asin', 'datetime', or 'processed_text'
df = df.dropna(subset=columns_to_check)

# Select relevant columns
df_processed = df[['Datetime', 'asin', 'processed_text']]

# Show the first 5 rows of the processed data
print("First 5 rows of processed data:")
print(df_processed.head())

# Save the cleaned and processed DataFrame to a new CSV file
df_processed.to_csv(output_file_path, index=False)

print(f"Processed file saved to {output_file_path}")

# Generate and display a word cloud
# Combine all processed text into one large string
all_text = ' '.join(df_processed['processed_text'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide the axes
plt.title('Word Cloud of Processed Text', fontsize=16)
plt.show()
