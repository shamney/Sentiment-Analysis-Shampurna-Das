import json
import csv

# Define input and output file paths
input_file = '/Users/shampurnadas/Desktop/Project/All_Beauty.jsonl'
output_file = '/Users/shampurnadas/Desktop/reviews.csv'

# Initialize a counter
counter = 0

# Open the input JSON Lines file and the output CSV file
with open(input_file, 'r') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    # Create a CSV writer object
    csv_writer = csv.writer(outfile)

    # Initialize header as None initially
    header = None

    # Read the JSONL file line-by-line
    for line in infile:
        if counter < 90000:  # Limit to the first 90,000 records

            record = json.loads(line)

            # Write the header only once, using the keys of the first record
            if header is None:
                header = list(record.keys())
                csv_writer.writerow(header)  # Write header to CSV

            # Write the values of the record as a row in the CSV file
            csv_writer.writerow(record.values())

            # Increment the counter
            counter += 1
        else:
            break  # Stop reading once we have 90,000 records

print(f"Successfully saved the first {counter} records to {output_file}.")
