from pickle import load
from flask import Flask
from flask import jsonify
from flask import request

app = Flask('predict_web')

# load logisticRegression model from pickle file
with open('model1.bin', 'rb') as output:
    model = load(output)

# load dictvectorizer for preprocess the data
with open('dv1.bin', 'rb') as d:
    dv = load(d)



@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[:, 1]
    client_credit_score = {
        'score': float(y_pred)
    }
    return jsonify(client_credit_score) 

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8686)


# predict new feature
customer = {"job": "retired", "duration": 445, "poutcome": "success"}

customer = dv.transform(customer)

credit_proba = model.predict_proba(customer)[:, 1]
print(credit_proba)
