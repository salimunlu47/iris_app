#!/usr/bin/env python
# coding: utf-8

# In[1]:

import seaborn as sns
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
iris_data = sns.load_dataset('iris')

# Preprocess the dataset
X = iris_data.drop('species', axis=1)
y = iris_data['species']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train the models
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier()
}

trained_models = {}
performance_metrics = {}

for model_name, model in models.items():
    # Calculate performance metrics using cross-validation
    cv_accuracy = cross_val_score(model, X, y, cv=5, scoring=make_scorer(accuracy_score)).mean()
    cv_precision = cross_val_score(model, X, y, cv=5, scoring=make_scorer(precision_score, average='weighted')).mean()
    cv_recall = cross_val_score(model, X, y, cv=5, scoring=make_scorer(recall_score, average='weighted')).mean()
    cv_f1 = cross_val_score(model, X, y, cv=5, scoring=make_scorer(f1_score, average='weighted')).mean()

    # Round the performance metrics to 4 decimal places
    cv_accuracy = round(cv_accuracy, 4)
    cv_precision = round(cv_precision, 4)
    cv_recall = round(cv_recall, 4)
    cv_f1 = round(cv_f1, 4)

    # Save the performance metrics
    performance_metrics[model_name] = {
        'accuracy': cv_accuracy,
        'precision': cv_precision,
        'recall': cv_recall,
        'f1_score': cv_f1
    }

    # Train the model on the entire dataset
    model.fit(X, y)
    trained_models[model_name] = model

# Save the trained models and performance metrics
joblib.dump(trained_models, 'trained_models.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
joblib.dump(performance_metrics, 'performance_metrics.joblib')


