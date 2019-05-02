import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import ast
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

def encode(original_dataset):
	# One-hot-encodes features and drops the original features from the dataset

    dummies = pd.get_dummies(original_dataset)
    return pd.concat([original_dataset, dummies], axis=1)

def feature_transform(df):
    '''
    Extract n features from json 
    Create n new columns
    ''' 
    if pd.isnull(df):
        return
    for feature in ast.literal_eval(df):
        dataset[feature["name"]] = 1
            
def release_date_transform(df):
    '''
    Transform release date into year and month
    '''
    return
    
def transform_features(dataset):
    dataset['genres'] = dataset['genres'].apply(feature_transform)
    dataset['production_companies'] = dataset['production_companies'].apply(feature_transform)
    dataset['production_countries'] = dataset['production_countries'].apply(feature_transform)
    dataset['spoken_languages'] = dataset['spoken_languages'].apply(feature_transform)
    dataset['cast'] = dataset['cast'].apply(feature_transform)
    # Convert month-day-year column to separate year and month columns
    dataset['release_date'] = pd.to_datetime(dataset['release_date']).apply(release_date_transform)

def preprocess(dataset):
	# Clean and process the data 
    transform_features(dataset)
    dropped_cols = [
        'id',
        'belongs_to_collection',
        'homepage',
        'imdb_id', 
        'overview',
        'status', 
        'tagline',
        'poster_path',
        'Keywords',
        'crew',
        'genres',
        'production_companies',
        'production_countries',
        'spoken_languages',
        'cast',
        'release_date'
    ]
    dataset = dataset.drop(dropped_cols, axis=1)
    print(dataset.columns)
    dataset.fillna(0)
    dataset = encode(dataset)
    return dataset

training_dataset = pd.read_csv('../input/train.csv')
X = preprocess(training_dataset.drop(columns='revenue'))
y = training_dataset['revenue']

Xtest = encode(pd.read_csv('../input/test.csv'))

model = LogisticRegression()
rfe = RFECV(model, cv=10, n_jobs=-1)
fit = rfe.fit(X, y)

print(rfe.support_)



