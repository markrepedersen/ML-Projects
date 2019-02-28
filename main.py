import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

def encode(original_dataset, feature_to_encode):
	#One-hot-encodes a feature and drops the original feature from the dataset

    dummies = pd.get_dummies(original_dataset[[feature_to_encode]])
    return pd.concat([original_dataset, dummies], axis=1).drop([feature_to_encode], axis=1)

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

y = train_data['Survived']
X = train_data.drop('Survived', axis=1)

model.fit(X,y)

y_pred = model.predict(X)
tr_error = np.mean(y_pred != y)

y_pred = model.predict(X_test)
te_error = np.mean(y_pred != y_test)
print("    Training error: %.3f" % tr_error)
print("    Testing error: %.3f" % te_error)
