import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


# READ DATA
df = pd.read_csv('adult.csv', sep=',')

# DROP USELESS VARIABLES

df = df.drop(['fnlwgt','education.num', 'occupation', 'relationship', 
        'capital.gain', 'capital.loss', 'native.country'], axis=1)


# SET INCOME AS BINARY VARIABLE
df['income'].replace(['<=50K', '>50K'],[0,1], inplace=True)

# delete the rows with missing data
df = df.loc[df['workclass'] != '?']

# split into depedent and independent
X = df.drop('income', axis=1)
y = df['income']

# Split X into continuous variables and categorical variables

X_continous = X[['age','hours.per.week']].reset_index(drop=True)

X_categorical = X[['workclass','education','marital.status','race','sex']].reset_index(drop=True)

# Fit One hot Encoder
enc = OneHotEncoder()
enc.fit(X_categorical)

# categorical data to one hot encoding
X_encoded = enc.transform(X_categorical).toarray()
X_encoded = pd.DataFrame(X_encoded)

# Concatenate both continous and encoded set
X = pd.concat([X_continous,X_encoded], axis=1)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3,stratify=y,random_state=10)

# Model
logit = LogisticRegression(max_iter=10000)
logit = logit.fit(X_train, y_train)



def predict_probability(age, workclass, education, marital_status, race, sex, hours):
    """This function predicts the probability of earning more than 50k a year, given some input variables"""
    encoded = enc.transform([[age, workclass, education, marital_status, race, sex]]).toarray()
    encoded = encoded.reshape(encoded.shape[1],)
    continous = np.array([age, hours])

    processed_data = np.concatenate((continous, encoded))
    processed_data = processed_data.reshape(1, processed_data.shape[0])
    prediction = logit.predict_proba(processed_data)[0][1]
    prediction = "{}%".format(round(prediction*100, 2))

    return prediction





