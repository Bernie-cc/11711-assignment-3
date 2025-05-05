'''
This file is used to process the data for the model.
Maintain only reviewerID, asin, overall, unixReviewTime
'''

import pandas as pd
import json

def load_json_file(file_path):
    """
    Load and read a JSON Lines file
    Args:
        file_path: path to the JSON Lines file
    Returns:
        data: list of loaded JSON objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def process_data(input_file, output_file):
    """
    Process the data and keep only required columns
    Args:
        input_file: path to input JSON file
        output_file: path to output CSV file
    """
    # Load data
    data = load_json_file(input_file)
    df = pd.DataFrame(data)
    
    # Keep only required columns
    required_columns = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
    df_processed = df[required_columns]
    
    # Save to CSV file
    df_processed.to_csv(output_file, index=False)
    
    print(f"Processing completed. Saved to: {output_file}")
    print(f"Data shape: {df_processed.shape}")
    
if __name__ == "__main__":
    input_file = "reviews_Beauty_5.json"
    output_file = "beauty_processed.csv"
    process_data(input_file, output_file)










