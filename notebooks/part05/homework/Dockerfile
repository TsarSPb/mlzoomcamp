FROM agrigorev/zoomcamp-model:3.8.12-slim

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "homework_predict_flask.py", "model1.bin", "dv.bin", "./"]

RUN pip install pipenv && pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "homework_predict_flask:app"]
