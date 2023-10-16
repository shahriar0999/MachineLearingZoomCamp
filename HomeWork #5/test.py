from pickle import load

# load logisticRegression model from pickle file
with open('model1.bin', 'rb') as output:
    model = load(output)

# load dictvectorizer for preprocess the data
with open('dv1.bin', 'rb') as d:
    dv = load(d)

# predict new feature
customer = {"job": "retired", "duration": 445, "poutcome": "success"}

customer = dv.transform(customer)

credit_proba = model.predict_proba(customer)[:, 1]
print(credit_proba)
