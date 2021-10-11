# Loading model
import pickle
input_model = f'model1.bin'
input_dv = f'dv.bin'

with open(input_model, 'rb') as f_in:
    model = pickle.load(f_in)
with open(input_dv, 'rb') as f_in:
    dv = pickle.load(f_in)

sample_data = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

X = dv.transform(sample_data)

y_pred = model.predict_proba(X)[0, 1]

print("Input: ", sample_data)
print("Churn prediction: ", y_pred)
