#!/usr/bin/env python
# coding: utf-8

import gzip
from collections import defaultdict
import numpy as np
from sklearn.metrics import jaccard_score
from tqdm.notebook import tqdm

# Utility Functions
def assertFloat(x):
    assert isinstance(float(x), float)

def assertFloatList(items, N):
    assert len(items) == N
    assert all(isinstance(float(x), float) for x in items)

def readGz(path):
    for line in gzip.open(path, 'rt'):
        yield eval(line)

def readJSON(path):
    with gzip.open(path, 'rt') as f:
        f.readline()  # Skip header
        for line in f:
            yield eval(line)

# Load Data
allHours = list(readJSON("../data/train.json.gz"))
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

# Data Preprocessing
df_train = pd.DataFrame(hoursTrain)
df_valid = pd.DataFrame(hoursValid)

# Question 1: Baseline Play Prediction
played_games = set(entry[1] for entry in allHours)

playedValid = [(u, g, 1) for u, g, _ in hoursValid]
notPlayedValid = [(u, random.choice(list(played_games - {g})), 0) for u, g, _ in hoursValid]

merged_valid_set = playedValid + notPlayedValid

gameCount = defaultdict(int)
totalPlayed = 0

for u, g, _ in allHours:
    gameCount[g] += 1
    totalPlayed += 1

mostPopular = sorted([(count, game) for game, count in gameCount.items()], reverse=True)

return1 = set()
count = 0
for ic, game in mostPopular:
    count += ic
    return1.add(game)
    if count > totalPlayed / 2:
        break

predictions = set()
for user, game, _ in merged_valid_set:
    pred = 1 if game in return1 else 0
    predictions.add((user, game, pred))

correct = sum(1 for pred in predictions if pred in merged_valid_set)
total_predictions = len(merged_valid_set)
accuracy = correct / total_predictions

answers = {'Q1': accuracy}
assertFloat(answers['Q1'])

# Question 2: Optimizing Threshold

def test_thresh(val):
    threshold = totalPlayed * (val / 100)
    return1 = set()
    count = 0
    for ic, game in mostPopular:
        count += ic
        return1.add(game)
        if count > threshold:
            break

    predictions = [(u, g, 1 if g in return1 else 0) for u, g, _ in merged_valid_set]
    correct = sum(1 for i, pred in enumerate(predictions) if pred == merged_valid_set[i])
    return correct / len(merged_valid_set)

top_accuracy = 0
top_thresh = 0
for i in tqdm(range(1, 101)):
    accuracy = test_thresh(i)
    if accuracy > top_accuracy:
        top_accuracy = accuracy
        top_thresh = i

answers['Q2'] = [top_thresh / 100, top_accuracy]
assertFloatList(answers['Q2'], 2)

# Question 6: Hours Played Prediction
trainHours = [entry[2]['hours_transformed'] for entry in hoursTrain]
globalAverage = np.mean(trainHours)

# Initialize parameters
alpha = globalAverage
lambda_reg = 1.0

users = list(set(u for u, _, _ in allHours))
items = list(set(g for _, g, _ in allHours))
user_to_index = {user: i for i, user in enumerate(users)}
item_to_index = {item: i for i, item in enumerate(items)}

num_users = len(users)
num_items = len(items)
beta_user = np.zeros(num_users)
beta_item = np.zeros(num_items)

# Training with Stochastic Gradient Descent
learning_rate = 0.01
num_epochs = 10
for epoch in range(num_epochs):
    for u, g, record in hoursTrain:
        user_idx = user_to_index[u]
        item_idx = item_to_index[g]

        pred = alpha + beta_user[user_idx] + beta_item[item_idx]
        error = record['hours_transformed'] - pred

        alpha += learning_rate * (error - lambda_reg * alpha)
        beta_user[user_idx] += learning_rate * (error - lambda_reg * beta_user[user_idx])
        beta_item[item_idx] += learning_rate * (error - lambda_reg * beta_item[item_idx])

# Evaluate on Validation Set
valid_preds = []
actuals = []
for u, g, record in hoursValid:
    user_idx = user_to_index[u]
    item_idx = item_to_index[g]

    pred = alpha + beta_user[user_idx] + beta_item[item_idx]
    valid_preds.append(pred)
    actuals.append(record['hours_transformed'])

validMSE = np.mean((np.array(valid_preds) - np.array(actuals)) ** 2)
answers['Q6'] = validMSE
assertFloat(answers['Q6'])

# Save Answers
with open("answers_hw3.txt", 'w') as f:
    f.write(str(answers) + '\n')
