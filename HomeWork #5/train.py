import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pickle


# preprocesing steps
bank = pd.read_csv('bank.csv',sep=';')
features = ['job','duration', 'poutcome']
dicts = bank[features].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(dicts)
y = bank.y

# train the model
model = LogisticRegression().fit(X, y)


with open('model1.bin', 'wb') as mod:
    pickle.dump(model, mod)

with open('dv1.bin', 'wb') as d_v:
    pickle.dump(dv, d_v)
