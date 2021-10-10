#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pickle
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, roc_auc_score, accuracy_score

from tqdm.auto import tqdm


# # Preparing data 
# 
# We'll talk about this dataset in more details in week 6. But for now, use the following code to get started

# In[2]:


df = pd.read_csv('../../data/CreditScoring.csv')
df.columns = df.columns.str.lower()


# Some of the features are encoded as numbers. Use the following code to de-code them:

# In[3]:


status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}
df.status = df.status.map(status_values)

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}
df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}
df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}
df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}
df.job = df.job.map(job_values)


# In[4]:


for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=0)


# Remove clients with unknown default status

# In[5]:


df = df[df.status != 'unk'].reset_index(drop=True)


# In[6]:


df['default'] = (df.status == 'default').astype(int)
del df['status']


# In[7]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.default.values
y_val = df_val.default.values
y_test = df_test.default.values

del df_train['default']
del df_val['default']
del df_test['default']


# In[8]:


numeric_columns = df.select_dtypes(include=[np.number]).columns
numeric_columns


# In[9]:


df[numeric_columns].head()


# In[10]:


for c in numeric_columns:
    print(c,"    \t",roc_auc_score(df['default'],df[c]).round(3))


# In[11]:


cols = ['seniority', 'income', 'assets', 'records', 'job', 'home']


# In[12]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[cols].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[13]:


def predict(df, dv, model):
    dicts = df[cols].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# # Saving model to pickle

# In[14]:


C=1
dv, model = train(df_full_train, df_full_train.default, C)


# In[15]:


y_pred = predict(df_test, dv, model)


# In[16]:


auc = roc_auc_score(y_test, y_pred)
auc


# In[17]:


output_file = f'model_C={C}.pkl'
output_file


# In[18]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model),f_out)


# # Loading model

# In[19]:


import pickle


# In[20]:


input_file = f'model_C=1.pkl'


# In[21]:


with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[22]:


model, dv


# In[23]:


df_test


# In[45]:


df_test.iloc[:2].to_json(orient='records')


# In[49]:


sample_data = {"seniority":14,"home":"parents","time":24,"age":19,"marital":"single","records":"no","job":"fixed","expenses":35,"income":28,"assets":0,"debt":0,"amount":400,"price":600}
# sample_data = df_test.iloc[:2].to_json(orient='records')


# In[50]:


X = dv.transform(sample_data)
X


# In[51]:


model.predict_proba(X)

