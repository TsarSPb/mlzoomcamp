from flask import Flask, jsonify, request
import pickle

input_model = f'model1.bin'
input_dv = f'dv.bin'
with open(input_model, 'rb') as f_in:
    model = pickle.load(f_in)
with open(input_dv, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods = ['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

@app.route('/ping', methods = ['GET'])
def ping():
    return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port = 9696)
