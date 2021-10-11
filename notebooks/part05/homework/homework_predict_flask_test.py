import requests

url = 'http://localhost:9696/predict'
customer = {"contract": "two_year", "tenure": 1, "monthlycharges": 10}

response = requests.post(url, json=customer).json()
print(response)
if response['churn'] is True:
    print('Churn is True, sending a prom oemail...')
else:
    print('Churn is False, no need to do anything...')
