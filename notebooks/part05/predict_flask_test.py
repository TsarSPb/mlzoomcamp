import requests

url = 'http://localhost:9696/predict'
customer = {"seniority": 14, "home": "parents", "time": 24, "age": 19, "marital": "single", "records": "no",
            "job": "fixed", "expenses": 35, "income": 28, "assets": 0, "debt": 0, "amount": 400, "price": 600}

response = requests.post(url, json=customer).json()
if response['churn'] == True:
    print('Churn is True, sending a prom oemail...')
else:
    print('Churn is False, no need to do anything...')
