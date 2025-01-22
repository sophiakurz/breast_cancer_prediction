#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, plot_importance


# In[10]:


data = pd.read_csv('breast-cancer-wisconsin-data_data.csv')
data.head()


# In[11]:


data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data = data.drop(columns=['id', 'Unnamed: 32'])
data.head()


# In[12]:


print("Columns in DataFrame:", data.columns)


# In[17]:


y = data['diagnosis']
X = data.drop("diagnosis", axis='columns')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[20]:


k_best_selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = k_best_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = k_best_selector.transform(X_test_scaled)

selected_features = X.columns[k_best_selector.get_support()]
print("Selected Features:", selected_features.tolist())


# In[28]:


model_params = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
    },
    "k-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss', random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    }
}


# In[29]:


results = {}

for model_name, mp in model_params.items():
    print(f"Tuning {model_name}...")
    grid_search = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_selected, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_selected)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    results[model_name] = {
        "Best Parameters": grid_search.best_params_,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }


# In[30]:


print("\nModel Performance Summary:")
for model, metrics in results.items():
    print(f"{model}:")
    for metric_name, metric_value in metrics.items():
        if metric_name == "Best Parameters":
            print(f"  {metric_name}: {metric_value}")
        else:
            print(f"  {metric_name}: {metric_value:.2f}")

