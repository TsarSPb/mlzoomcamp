# Loading model
import pickle
input_file = f'model_C=1.pkl'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

sample_data = {"seniority":14,"home":"parents","time":24,"age":19,"marital":"single","records":"no","job":"fixed","expenses":35,"income":28,"assets":0,"debt":0,"amount":400,"price":600}

X = dv.transform(sample_data)

y_pred = model.predict_proba(X)[0, 1]

print("Input: ", sample_data)
print("Churn prediction: ", y_pred)