import gzip
from collections import defaultdict
import math
import random
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yake

# Function to read JSON data
def readJSON(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            data = eval(line)
            yield data['userID'], data['gameID'], data

# Preprocess the training data
allHours = list(readJSON("./../data/train.json.gz"))
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

# Data structures for efficient lookup
hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)
for user, game, data in hoursTrain:
    hours = data['hours_transformed']
    hoursPerUser[user].append((game, hours))
    hoursPerItem[game].append((user, hours))

# Jaccard similarity calculation
def Jaccard(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# Extract topics using YAKE
kw_extractor = yake.KeywordExtractor()
user_review_text = defaultdict(list)
game_review_text = defaultdict(list)

for user, game, data in allHours:
    user_review_text[user].append(data['text'])
    game_review_text[game].append(data['text'])

user_topics = {
    user: [kw[0] for kw in kw_extractor.extract_keywords(' '.join(texts))[:10]]
    for user, texts in tqdm(user_review_text.items(), desc="Extracting user topics")
}
game_topics = {
    game: [kw[0] for kw in kw_extractor.extract_keywords(' '.join(texts))[:10]]
    for game, texts in tqdm(game_review_text.items(), desc="Extracting game topics")
}

# Prepare predictions for 'Played'
predictions_played = []
for line in tqdm(open("./../data/pairs_Played.csv"), desc="Predicting 'Played')"):
    if line.startswith("userID"):
        predictions_played.append(line.strip())
        continue

    user, game = line.strip().split(',')
    max_similarity = max(
        (Jaccard(set(hoursPerItem[game]), set(hoursPerItem[other_game])) for other_game, _ in hoursPerUser[user]),
        default=0
    )

    topic_similarity = Jaccard(set(user_topics.get(user, [])), set(game_topics.get(game, [])))
    prediction = 1 if (
        max_similarity > 0.19 or len(hoursPerItem[game]) > 62 or topic_similarity > 0.4
    ) or (
        max_similarity > 0.13 and topic_similarity > 0.3
    ) else 0
    predictions_played.append(f"{user},{game},{prediction}")

with open("predictions_Played.csv", 'w') as f:
    f.write('\n'.join(predictions_played))

# Baseline variables for 'Hours'
alpha = np.mean([data['hours_transformed'] for _, _, data in hoursTrain])
betaU = defaultdict(float)
betaI = defaultdict(float)

# Optimization for 'Hours'
def iterate(alpha, betaU, betaI, lamb=4.0):
    for user, games in hoursPerUser.items():
        betaU[user] = sum(
            hours - (alpha + betaI[game]) for game, hours in games
        ) / (len(games) + lamb)

    for game, users in hoursPerItem.items():
        betaI[game] = sum(
            hours - (alpha + betaU[user]) for user, hours in users
        ) / (len(users) + lamb)

    return sum(
        (alpha + betaU[user] + betaI[game] - hours) ** 2
        for user, game, hours in (
            (user, game, data['hours_transformed']) for user, game, data in hoursTrain
        )
    )

# Training the model
for _ in tqdm(range(100), desc="Optimizing parameters"):
    iterate(alpha, betaU, betaI)

# Prepare predictions for 'Hours'
predictions_hours = []
for line in open("./../data/pairs_Hours.csv"):
    if line.startswith("userID"):
        predictions_hours.append(line.strip())
        continue

    user, game = line.strip().split(',')
    prediction = alpha + betaU[user] + betaI[game]
    predictions_hours.append(f"{user},{game},{prediction}")

with open("predictions_Hours.csv", 'w') as f:
    f.write('\n'.join(predictions_hours))
