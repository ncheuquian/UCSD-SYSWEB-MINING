import json
import gzip
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, confusion_matrix, balanced_accuracy_score, precision_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize answers dictionary
answers = {}

# Helper functions
def assert_float(x):
    assert isinstance(float(x), float)

def assert_float_list(items, N):
    assert len(items) == N
    assert all(isinstance(float(x), float) for x in items)

def load_json_gz(filepath):
    with gzip.open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def preprocess_dates(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d %H:%M:%S%z', utc=True)
    df['weekday'] = df[date_column].dt.strftime('%A')
    df['month'] = df[date_column].dt.strftime('%B')
    return df

def drop_unused_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop, errors='ignore')

# Load fantasy dataset
fantasy_data = load_json_gz("./../data/fantasy_10000.json.gz")

# Precompute max review length
max_review_length = max(len(d['review_text']) for d in fantasy_data)

# Feature function
def feature(datum):
    return len(datum['review_text']) / max_review_length

# Question 1: Train Linear Regression Model
X = [[1, feature(d)] for d in fantasy_data]
Y = [d['rating'] for d in fantasy_data]
reg = linear_model.LinearRegression().fit(X, Y)
answers['Q1'] = [reg.coef_[0], reg.coef_[1], mean_squared_error(Y, reg.predict(X))]
assert_float_list(answers['Q1'], 3)

# Question 2: One-Hot Encoding
fantasy_df = preprocess_dates(pd.DataFrame(fantasy_data), 'date_added')
columns_to_drop = ['user_id', 'book_id', 'review_id', 'date_added', 'date_updated', 
                   'read_at', 'started_at', 'n_votes', 'n_comments', 'review_text']
fantasy_df['review_text_prop'] = fantasy_df['review_text'].apply(lambda x: len(x) / max_review_length)
one_hot_df = pd.get_dummies(fantasy_df, columns=['weekday', 'month'], drop_first=True)
fantasy_df = drop_unused_columns(fantasy_df, columns_to_drop)

# Prepare features
X_one_hot = one_hot_df.drop(columns=['rating'])
Y_one_hot = one_hot_df['rating']
answers['Q2'] = [np.append(X_one_hot.iloc[0].values, 1), np.append(X_one_hot.iloc[1].values, 1)]
assert_float_list(answers['Q2'][0], len(X_one_hot.columns) + 1)

# Question 3: Compare MSE for direct and one-hot encoding
label_encoder = LabelEncoder()
fantasy_df['weekday'] = label_encoder.fit_transform(fantasy_df['weekday'])
fantasy_df['month'] = label_encoder.fit_transform(fantasy_df['month'])

X_direct = fantasy_df.drop(columns=['rating'])
Y_direct = fantasy_df['rating']

reg_direct = linear_model.LinearRegression().fit(X_direct, Y_direct)
reg_one_hot = linear_model.LinearRegression().fit(X_one_hot, Y_one_hot)

answers['Q3'] = [mean_squared_error(Y_direct, reg_direct.predict(X_direct)),
                 mean_squared_error(Y_one_hot, reg_one_hot.predict(X_one_hot))]
assert_float_list(answers['Q3'], 2)

# Question 4: Train/Test Split
train_size = len(X_direct) // 2
X_train_direct, X_test_direct = X_direct[:train_size], X_direct[train_size:]
X_train_one_hot, X_test_one_hot = X_one_hot[:train_size], X_one_hot[train_size:]
Y_train, Y_test = Y_direct[:train_size], Y_direct[train_size:]

reg_direct = linear_model.LinearRegression().fit(X_train_direct, Y_train)
reg_one_hot = linear_model.LinearRegression().fit(X_train_one_hot, Y_train)

answers['Q4'] = [mean_squared_error(Y_test, reg_direct.predict(X_test_direct)),
                 mean_squared_error(Y_test, reg_one_hot.predict(X_test_one_hot))]
assert_float_list(answers['Q4'], 2)

# Question 5: Logistic Regression for Beer Data
beer_data = load_json_gz("./../data/beer_50000.json.gz")
beer_df = pd.DataFrame(beer_data)
beer_df['binarized_rating'] = (beer_df['review/overall'] >= 4).astype(int)
beer_df['review_length'] = beer_df['review/text'].str.len()

X_beer = beer_df[['review_length']]
Y_beer = beer_df['binarized_rating']
log_reg = linear_model.LogisticRegression(class_weight='balanced').fit(X_beer, Y_beer)

Y_pred = log_reg.predict(X_beer)
conf_matrix = confusion_matrix(Y_beer, Y_pred)
TN, FP, FN, TP = conf_matrix.ravel()
answers['Q5'] = [TP, TN, FP, FN, 1 - balanced_accuracy_score(Y_beer, Y_pred)]
assert_float_list(answers['Q5'], 5)

# Question 6: Precision at K
K_values = [1, 100, 1000, 10000]
Y_prob = log_reg.predict_proba(X_beer)[:, 1]
sorted_indices = np.argsort(-Y_prob)
precs = []
for K in K_values:
    top_K_indices = sorted_indices[:K]
    precision = precision_score(Y_beer.iloc[top_K_indices], (Y_prob[top_K_indices] >= 0.5).astype(int))
    precs.append(precision)

plt.plot(K_values, precs, marker='o', linestyle='-')
plt.xscale('log')
plt.xlabel('K (log scale)')
plt.ylabel('Precision at K')
plt.title('Precision at K')
plt.grid(True)
plt.show()
answers['Q6'] = precs
assert_float_list(answers['Q6'], len(K_values))

# Write answers to file
with open("answers_hw1.txt", 'w') as f:
    f.write(str(answers) + '\n')
