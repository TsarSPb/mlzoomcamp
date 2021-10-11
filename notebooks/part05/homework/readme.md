# Question 1
`python3 install pipenv`  
The answer is in the console output:
```
Successfully installed backports.entry-points-selectable-1.1.0 distlib-0.3.3 filelock-3.3.0 importlib-metadata-4.8.1 importlib-resources-5.2.2 pip-21.2.4 pipenv-2021.5.29 platformdirs-2.4.0 typing-extensions-3.10.0.2 virtualenv-20.8.1 virtualenv-clone-0.5.7 zipp-3.6.0
```

# Question 2
pipenv install numpy scikit-learn flask  
The answer is in `Pipfile.lock` file:
```
...
        "scikit-learn": {
            "hashes": [
                "sha256:121f78d6564000dc5e968394f45aac87981fcaaf2be40cfcd8f07b2baa1e1829",
...
```

# Question 3
```
(part05) root@tsarev:/mnt/d/DOC_my/YandexDrive/Code/MLZoomcamp/notebooks/part05/homework# python homework_predict.py
Input:  {'contract': 'two_year', 'tenure': 12, 'monthlycharges': 19.7}
Churn prediction:  0.11549580587832914
```

# Question 4
```
>python .\homework_predict_flask_test.py
{'churn': True, 'churn_probability': 0.9988892771007961}
Churn is True, sending a prom oemail...
```

# Question 5
```
(part05) root@tsarev:/mnt/d/DOC_my/YandexDrive/Code/MLZoomcamp/notebooks/part05/homework# docker image ls --digests | grep agrigoriev
REPOSITORY                                 TAG                                                     DIGEST
               IMAGE ID       CREATED         SIZE
agrigorev/zoomcamp-model                   3.8.12-slim                                             sha256:1ee036b365452f8a1da0dbc3bf5e7dd0557cfd33f0e56b28054d1dbb9c852023   f0f43f7bc6e0   6 days ago      122MB
```

# Question 6
```
(mlzoomcamp) d:\DOC_my\YandexDrive\Code\MLZoomcamp\notebooks\part05\homework>python homework_predict_docker_test.py
{'churn': False, 'churn_probability': 0.32940789808151005}
Churn is False, no need to do anything...
```