# Links
[Zoomcamp homepage](https://datatalks.club/courses/2021-winter-ml-zoomcamp.html)  
[GitHub repo](https://github.com/alexeygrigorev/mlbookcamp-code)  
[Slack channel](https://datatalks-club.slack.com)  
[Youtube ML Zoomcamp](https://www.youtube.com/channel/UCDvErgK0j5ur3aLgn6U-LqQ)  

# Preparing the environment
```
python -m pip install --upgrade pip
python -m venv d:\opt\python_environments\mlzoomcamp
d:\opt\python_environments\mlzoomcamp\Scripts\Activate.ps1
pip install numpy pandas scikit-learn seaborn jupyter
git config --list --show-origin
git init .
git remote add origin https://github.com/TsarSPb/mlzoomcamp.git
git pull origin main
git branch -a
git checkout main
git add .
git commit -m "erster Commit. Richte mich ein."
git push origin main
```

# Session 20210909
## Lesson 1.4 CRISP-DM
<img src="images/CRISPDM.jpg" width="300" />  

It's about framing the big picture.
6 steps:
1. Business understanding  
Understand the scope of the problem and if ML is a good fit for this
1. Data understanding  
Is it available, what's there in the data, is there enough data
1. Data preparation  
Cleaning, converting, building pipelines
1. Modeling  
Building ML models, feature engineering
1. Evaluation  
Reevaluate with the business, check the metrics
1. Deploy  
Monitor and evaluate

Rinse and repeat.

## Lesson 1.7 Numpy refresher
```
python -m ipykernel install --user --name mlzoomcamp
jupyter kernelspec list
jupyter notebook --port 8888
```

