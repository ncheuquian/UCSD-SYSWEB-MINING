#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import matplotlib.pyplot as plt
import gzip
import json
from tqdm import tqdm
from datetime import date

# Utility Functions
def parse_json_gzip(filepath):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        return json.load(f)

def apply_features(df, cat_id=None, max_review_length=None):
    """Feature Engineering Function."""
    features = pd.DataFrame()
    if cat_id:
        features['beer/style'] = df['beer/style'].apply(lambda x: cat_id.get(x, "dummy"))
        features = pd.get_dummies(features, columns=['beer/style']).drop(columns=['beer/style_dummy'])
    features['review/aroma'] = df['review/aroma']
    features['review/appearance'] = df['review/appearance']
    features['review/palate'] = df['review/palate']
    features['review/taste'] = df['review/taste']
    features['review/overall'] = df['review/overall']
    if max_review_length:
        features['review_length'] = df['review/text'].apply(lambda x: len(x) / max_review_length)
    return features

def train_model(X_train, y_train, C=1.0, class_weight=None):
    """Train Logistic Regression Model."""
    model = LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def calculate_ber(y_true, y_pred):
    """Calculate Balanced Error Rate (BER)."""
    return 1 - balanced_accuracy_score(y_true, y_pred)

# Question 1: Logistic Regression
# Load and Prepare Data
data_path = "path_to_data.json"
data = pd.DataFrame(parse_json_gzip(data_path))
random.seed(0)
data = shuffle(data)

train_size = int(len(data) * 0.5)
val_size = int(len(data) * 0.25)
test_size = len(data) - train_size - val_size

data_train = data[:train_size]
data_val = data[train_size:train_size + val_size]
data_test = data[train_size + val_size:]

max_review_length = max(data['review/text'].apply(len))

# Feature Engineering
X_train = apply_features(data_train, max_review_length=max_review_length)
X_val = apply_features(data_val, max_review_length=max_review_length)
X_test = apply_features(data_test, max_review_length=max_review_length)

y_train = data_train['label']
y_val = data_val['label']
y_test = data_test['label']

# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Train Logistic Regression
model = train_model(X_train, y_train, C=1.0)
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

train_ber = calculate_ber(y_train, y_pred_train)
val_ber = calculate_ber(y_val, y_pred_val)
test_ber = calculate_ber(y_test, y_pred_test)

print(f"Q1 Train BER: {train_ber}, Validation BER: {val_ber}, Test BER: {test_ber}")

# Question 2: Class-Weighted Logistic Regression
model_balanced = train_model(X_train, y_train, C=1.0, class_weight='balanced')
y_pred_balanced_val = model_balanced.predict(X_val)
y_pred_balanced_test = model_balanced.predict(X_test)

val_ber_balanced = calculate_ber(y_val, y_pred_balanced_val)
test_ber_balanced = calculate_ber(y_test, y_pred_balanced_test)

print(f"Q2 Validation BER: {val_ber_balanced}, Test BER: {test_ber_balanced}")

# Question 3: Regularization Pipeline
C_values = [10**i for i in range(-4, 5)]
val_bers = []

for C in C_values:
    model = train_model(X_train, y_train, C=C, class_weight='balanced')
    y_pred_val = model.predict(X_val)
    val_ber = calculate_ber(y_val, y_pred_val)
    val_bers.append(val_ber)

best_C = C_values[np.argmin(val_bers)]
model_best = train_model(X_train, y_train, C=best_C, class_weight='balanced')
y_pred_test_best = model_best.predict(X_test)

test_ber_best = calculate_ber(y_test, y_pred_test_best)

print(f"Q3 Best C: {best_C}, Best Test BER: {test_ber_best}")

# Question 4: Jaccard Similarity
# Load Goodreads Dataset
goodreads_path = "path_to_young_adult.json.gz"
goodreads_data = parse_json_gzip(goodreads_path)

item_sets = {item['book_id']: set(item['users']) for item in goodreads_data}
first_item = list(item_sets.keys())[0]

similarities = []
for item_id, users in item_sets.items():
    if item_id != first_item:
        sim = 1 - len(item_sets[first_item].intersection(users)) / len(item_sets[first_item].union(users))
        similarities.append((sim, item_id))

similarities = sorted(similarities, reverse=True)[:10]

print(f"Q4 Top 10 Similarities: {similarities}")

# Save Results
answers = {
    "Q1": [train_ber, val_ber, test_ber],
    "Q2": [val_ber_balanced, test_ber_balanced],
    "Q3": [best_C, test_ber_best],
    "Q4": similarities
}

with open("answers_hw2.txt", "w") as f:
    json.dump(answers, f, indent=4)

print("Answers saved to answers_hw2.txt.")
