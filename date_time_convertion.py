import pandas as pd
from datetime import datetime

# Step 1: Define the file path for the input file on the desktop
input_file_path = '~/Desktop/reviews.csv'  #  reviews.csv file path
output_file_path = '~/Desktop/updatedreviews.csv'  # Path for the output file

# Step 2: Read the CSV file
df = pd.read_csv(input_file_path)

# Step 3: Convert the "timestamp" column to datetime format
df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert Unix time (ms) to human-readable format

# Step 4: Save the updated DataFrame to a new CSV file
df.to_csv(output_file_path, index=False)

print(f"Updated CSV file with datetime saved to: {output_file_path}")
