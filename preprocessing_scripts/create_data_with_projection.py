#!/usr/bin/env python
# coding: utf-8

# ## Create directory with cdrs
# !pip install --ignore-installed scikit-learn==0.21.3

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import joblib
import sklearn
from tqdm import tqdm, notebook
from typing import List

def load_object(path):
    with open(path, "rb") as file:
        output = pickle.load(file)
    return output    

def moving_window(sequence):
    """
        split squense to three components
    """
    corpus = []
    doc = [sequence[i:i + 3] for i in range(len(sequence) - 2)]
    corpus.append(' '.join(doc))
    return corpus
def convert_array2str(x:List, projection: object) -> str:
    arr = projection.transform(x).flatten()
    convert_arr2list = list(map(str, arr))
    return ','.join(convert_arr2list)

projection = joblib.load('binary_files_pickles/projection_tfidf.joblib')
v_gene = load_object('binary_files_pickles/v_gene_map.pkl')
j_gene = load_object('binary_files_pickles/j_gene_map.pkl')


# Test Pickles
# ```Python
# print(projection)
# print(v_gene)
# print(j_gene)
# ```

# Test Projections
# ```Python
# test_1 = ['CSV SVE VEE']
# projection = joblib.load('binary_files_pickles/projection_tfidf.joblib')
# projection.transform(test_1)
# ```

def create_projections_tables(path, projection, v_gene, j_gene):
    df = pd.read_csv(path)
        
    df_combined = df['combined'].str.split('_', expand=True)
    df_combined[1] = df_combined[1].apply(moving_window)
    df_combined[1] = df_combined[1].apply(lambda x: convert_array2str(x, projection)) 
    df_combined[2] = df_combined[2].map(j_gene)
    df_combined[0] = df_combined[0].map(v_gene)
    
    df['projection']=df_combined.apply(lambda x:'%s_%s_%s' % (x[0],x[1],x[2]),axis=1)
        
    return df
    

def project_files(input_directory, output_directory):
    for path in notebook.tqdm(input_directory.glob("*"), total=len(list(input_directory.glob("*")))):

        df = create_projections_tables(path, projection, v_gene, j_gene)
   
        name = '_'.join([path.stem, 'tfidf.csv'])

        destination = output_directory / name

        df.to_csv(destination, index=False)


# directory of data to project
input_directory = Path('/home/vovaps/Projects/EmersonSoftwareFramework/Data/PreprocessedTrain_641')
output_directory = Path('/home/vovaps/Projects/EmersonSoftwareFramework/Data/PreprocessedTrain_TFIDF')
project_files(input_directory, output_directory)

# directory of data to project
input_directory = Path('/home/vovaps/Projects/EmersonSoftwareFramework/Data/PreprocessedTest')
output_directory = Path('/home/vovaps/Projects/EmersonSoftwareFramework/Data/PreprocessedTest_TFIDF')
project_files(input_directory, output_directory)

# convert to string and comeback
# 
# ```Python
# ggg = convert_array2str(temp[0], projection)
# print(ggg)
# 
# ddd = np.array(list(map(float, ggg.split(','))))
# ddd
# ```

# In[ ]:




