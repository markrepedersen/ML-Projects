import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

def encode(original_dataset, features_to_encode):
	# One-hot-encodes features and drops the original features from the dataset

    dummies = pd.get_dummies(original_dataset[features_to_encode])
    return pd.concat([original_dataset, dummies], axis=1).drop(features_to_encode, axis=1)

def get_title_from_name(name):
	# Gets the title such as 'Mr', 'Mrs', 'Master', etc from a name and replace the name with the title

	if 'Mr' in name:
		return 'Mr'
	elif 'Mrs' in name:
		return 'Mrs'
	elif 'Ms' in name:
		return 'Mrs'
	elif 'Miss' in name:
		return 'Mrs'
	elif 'Mme' in name:
		return 'Mrs'
	elif 'Mlle' in name:
		return 'Mrs'
	elif 'Countess' in name:
		return 'Mrs'
	elif 'Master' in name:
		return 'Master'
	elif 'Dr' in name:
		return 'Master'
	elif 'Don' in name:
		return 'Master'
	elif 'Capt' in name:
		return 'Mr'
	elif 'Rev' in name:
		return 'Mr'
	elif 'Col' in name:
		return 'Mr'
	elif 'Jonkheer' in name:
		return 'Mr'
	elif 'Major' in name:
		return 'Mr'
	else:
		raise ValueError(name) 

def preprocess(dataset):
	# Clean and process the data 

	dataset = dataset.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1)
	dataset['Name'] = dataset['Name'].apply(get_title_from_name)
	dataset = encode(dataset, ['Name', 'Sex', 'Embarked'])
	average_nan_features(dataset)
	return dataset

def average_nan_features(dataset):
	dataset['Age'].fillna((dataset['Age'].mean()), inplace=True)
	dataset['Fare'].fillna((dataset['Fare'].mean()), inplace=True)

def evaluate(model, X, y, X_test):
	model.fit(X,y)

	y_pred = model.predict(X)
	tr_error = np.mean(y_pred != y)
	print("Training error: %.3f" % tr_error)

	return model.predict(X_test)

train_data = preprocess(pd.read_csv('./data/train.csv'))
test_csv = pd.read_csv('./data/test.csv')
X_test = preprocess(test_csv)

y = train_data['Survived']
X = train_data.drop('Survived', axis=1)

model = RandomForestClassifier(max_depth=100, n_estimators=100)

test_csv['Survived'] = evaluate(model, X, y, X_test)

test_csv[['PassengerId', 'Survived']].to_csv('prediction.csv', encoding='utf-8', index=False)
