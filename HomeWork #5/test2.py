import requests

url = "http://192.168.43.80:8686/predict"
# client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
client = {"job": "retired", "duration": 445, "poutcome": "success"}
score = requests.post(url, json=client).json()

print("Credit Score",score['score'])