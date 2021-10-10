FROM python:3.8.12-slim

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "predict_flask.py", "model_C=1.pkl", "./"]

RUN pip install pipenv && pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_flask:app"]
