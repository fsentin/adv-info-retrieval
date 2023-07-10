## implement part 1 here

"""
The flow of our approach is as follows:
- convert the raw judgements so that each judgement consists of query ID, passage ID, annotator ID, judgement value and ...?
- initialize the algorithm by assigning an initial probability distribution over the possible labels for each query-passage pair
- compute the expected labels for each query-passage pair based on the current probability distributions and the agreement among annotators and update the probability distributions based on the expected labels and the raw judgments until convergence
- once the algorithm converges, assign the label with the highest probability as the final aggregated label for each query-passage pair.
"""

import numpy as np
import pandas as pd


def convert_raw_judgements():
    """
    Convert the raw judgements so that each judgement consists of query ID, passaage ID, annotator ID, judgement value and ...?
    Add "true" relevance label to the dataset based on the baseline model.
    """
    ds = pd.read_csv("Part-1\\fira-22.judgements-anonymized.tsv", sep='\t')
    ds = ds.drop(columns=['id', 'relevanceCharacterRanges', 'durationUsedToJudgeMs', 'judgedAtUnixTS'])
    ds = ds.rename(columns={'queryId': 'query_id', 'documentId': 'passage_id', 'userId': 'annotator_id', 'relevanceLevel': 'judgement'})
    ds['judgement'] = ds['judgement'].apply(lambda x: int(x.split('_')[0]))

    # add "true" relevance label to the dataset based on the baseline model
    # load the baseline model
    baseline_model = pd.read_csv("Part-1\\fira-22.baseline-qrels.tsv", sep=' ', header=None)
    baseline_model.drop(columns=[1], inplace=True)
    baseline_model.rename(columns={0: 'query_id', 2: 'passage_id', 3: 'relevance'}, inplace=True)

    # merge the baseline model with the dataset
    ds = pd.merge(ds, baseline_model, how='left', on=['query_id', 'passage_id'])
    return ds

def confusion_matrix(userId, df):
    """
    Compute the confusion matrix for a given annotator.
    """
    # filter the dataset to get the judgements of the given annotator
    annotator_df = df[df['annotator_id'] == userId]
    # compute the confusion matrix
    confusion_matrix = pd.crosstab(annotator_df['judgement'], annotator_df['relevance'], rownames=['judgement'], colnames=['relevance'])
    # add missing columns and rows to the confusion matrix
    for i in range(4):
        if i not in confusion_matrix.index:
            confusion_matrix.loc[i] = 0.0
    for i in range(4):
        if i not in confusion_matrix.columns:
            confusion_matrix[i] = 0.0
    # sort the rows and columns of the confusion matrix
    confusion_matrix.sort_index(axis=0, inplace=True)
    confusion_matrix.sort_index(axis=1, inplace=True)
    # normalize the confusion matrix
    for i in range(4):
        if confusion_matrix.iloc[i].sum() != 0:
            confusion_matrix.iloc[i] = confusion_matrix.iloc[i] / confusion_matrix.iloc[i].sum()

    return confusion_matrix

ds = convert_raw_judgements()
# calculate the confusion matrix for each annotator and store it in a dictionary
confusion_matrices = {}
for userId in ds['annotator_id'].unique():
    confusion_matrices[userId] = confusion_matrix(userId, ds)

# initialize the probabilities of the relevance labels for each query-passage pair
newds = ds.drop_duplicates(subset=['query_id', 'passage_id'])
newds = newds.drop(columns=['annotator_id', 'judgement', 'relevance'])
#initialize the probability column for classes 0, 1, 2, 3
default_probabilities = {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0}
newds = newds.assign(**default_probabilities)

convergence = 100

# while convergence > 0.001:
# calculate the total probability of each label
total = newds['0'].sum()

# iterate over the dataset and calculate the probability of each label for each query-passage pair
for index, row in ds.iterrows():    
    # get the annotator ID
    userId = row['annotator_id']
    # get the confusion matrix of the annotator
    confusion_matrix = confusion_matrices[userId]
    # get the judgement value
    judgement = row['judgement']
    query_id = row['query_id']
    passage_id = row['passage_id']
    
    total = newds['0'].sum()

    # get row of the confusion matrix for the judgement value
    probabilities = confusion_matrix.iloc[judgement]
    newds.loc[(newds['query_id'] == query_id) & (newds['passage_id'] == passage_id), '0'] += probabilities[0]
    newds.loc[(newds['query_id'] == query_id) & (newds['passage_id'] == passage_id), '1'] += probabilities[1]
    newds.loc[(newds['query_id'] == query_id) & (newds['passage_id'] == passage_id), '2'] += probabilities[2]
    newds.loc[(newds['query_id'] == query_id) & (newds['passage_id'] == passage_id), '3'] += probabilities[3]

# normalize the probabilities
for index, row in newds.iterrows():
    total = row['0'] + row['1'] + row['2'] + row['3']
    newds.loc[index, '0'] = row['0'] / total
    newds.loc[index, '1'] = row['1'] / total
    newds.loc[index, '2'] = row['2'] / total
    newds.loc[index, '3'] = row['3'] / total
    # print(total)
    # print(newds.loc[index, '0'])

# calculate the convergence    
# total_new = newds['0'].sum()
# convergence = abs(total_new - total)
# update the judgements based on the new probabilities
# newds['judgement'] = newds[['0', '1', '2', '3']].idxmax(axis=1)
    

# assign the label with the highest probability as the final aggregated label for each query-passage pair
newds['final_label'] = newds[['0', '1', '2', '3']].idxmax(axis=1)
# keep only the query ID, passage ID and final label columns
newds = newds[['query_id', 'passage_id', 'final_label']]
# save the dataset
newds.to_csv("Part-1\\fira-22.judgements-aggregated.tsv", sep='\t', index=False)
