'''
This file is used to generate the train and test data for the model.
For each user, we can get the history interaction data from the csv file.
we use the last interaction as the test data and the second last interaction as the validation data and the rest as the train data.
Since in our project, we do not need to train any model, we can only preserve train and test data.

The input of this file is processed data from process_data.py
The output of this file is train.csv and test.csv

train.csv can be derived from the processed data by removing the last 2 interactions for each user. And it will 
preserved the same format as the processed data.

test.csv can be derived from the processed data by only keeping the last 1 interaction for each user. And it will 
preserved the same format as the processed data.
'''

import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os
def split_train_test(input_file, train_output, test_output, test_mode = False):
    """
    Split data into train and test sets based on user interactions
    Test data is the last interaction for each user
    Train data is the interaction except the last 2 for each user

    Example:
    If user 1 has interactions: [1, 2, 3, 4, 5]
    then the train data is [1, 2, 3] and the test data is [5]

    Args:
        input_file: path to processed CSV file from process_data.py
        train_output: path to save train data
        test_output: path to save test data
        test_mode: if True, only keep the first 1000 interactions for each user and delete the saved train and test files
    
    Returns:
        None
    """
    # Read the processed data
    print("Reading data...")
    df = pd.read_csv(input_file)

    if test_mode:
        df = df.head(1000)
    
    
    # Sort by timestamp for each user
    print("Sorting data...")
    df = df.sort_values(['reviewerID', 'unixReviewTime'])
    
    # Group by user and get indices for train and test sets
    train_indices = []
    test_indices = []
    
    print("Processing users...")
    for user, group in tqdm(df.groupby('reviewerID'), desc="Splitting data"):
        
        indices = group.index.tolist()

        if len(indices) > 2:  # Only include users with at least 3 interactions
            if test_mode:
                print(user, len(indices))
            # Last interaction for test
            test_indices.append(indices[-1])
            # second last interaction for validation but not used in this assignment    
            train_indices.extend(indices[:-2])
    
    # Split the data
    print("Creating train/test sets...")
    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]
    
    # Save to files
    print("Saving files...")
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    # Print statistics
    print("\nStatistics:")
    print(f"Total users: {df['reviewerID'].nunique()}")
    print(f"Total interactions: {len(df)}")
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # if test mode, delete the train and test files
    if test_mode:
        # get the input from user to confirm the deletion
        confirm = input("Are you sure you want to delete the train and test files? (y/n)")
        if confirm == "y":
            os.system("rm " + train_output)
            os.system("rm " + test_output)
            print("Train and test files deleted")
        else:
            print("Train and test files not deleted")

if __name__ == "__main__":
    input_file = "beauty_processed.csv"
    train_output = "beauty.train.csv"
    test_output = "beauty.test.csv"
    split_train_test(input_file, train_output, test_output, test_mode = False)






