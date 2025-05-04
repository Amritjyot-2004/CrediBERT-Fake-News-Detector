import requests

url = "http://localhost:5000/predict_api"
r = requests.post(url, json = {'headline':"Donald Trump’s $5 million 'gold card' visa: What it does mean for Indian nationals?"})

print(r.json())