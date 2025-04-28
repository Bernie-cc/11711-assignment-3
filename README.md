# 11711-assignment-4

## Overview

This assignment is to build a model to predict the rating of a product based on the user's history interaction data.
- We will first use nearest neignborhood method to coarse retrival the condidate products.
- We will then construct semantic and collaborative realtionship matrices for fine retrival by computing the retrival score
- Finally, we will leveage LLM for reranking the fine retrival candidate.

## Data
For this project, we utilize the Amazon Beauty dataset from 2015.  
You can find it at: https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews

## Data Process
After downloading the data, you can run following code to process data and generate the train test data. Since our recommendation system do not need training, we do not need valid dataset here. 
```bash
python process_data.py
python generate_train_test.py
python filter_meta_beauty.py
```

## Construct Matrices
```bash
python construct_collaborative.py
python embedding.py
python embedding_dict.py
``` 

## Experiment for retrival
```bash
python retrival_experiment.py
```

## Experiment for reranking
```bash
python rank_prompt_generation.py
python ranking.py
python evaluate_rank.py
```







