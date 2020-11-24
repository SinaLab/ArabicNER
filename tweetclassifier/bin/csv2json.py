import csv
import json
import random
import numpy as np
import os

csv_file = "/mnt/efs/data/twitter_us_airline_sentiment/twitter_us_airline_sentiment.csv"
output_path = "/mnt/efs/data/twitter_us_airline_sentiment"
tweets = list()
train_ratio = 0.7
dev_ratio = 0.1

with open(csv_file, "r") as fh:
    rows = csv.reader(fh)
    next(rows)

    for row in rows:
        tweets.append({
            "tweet": row[10],
            "labels": [row[1]]
        })

positive_tweets = [t for t in tweets if t["labels"][0] == "positive"]

negative_tweets = [t for t in tweets if t["labels"][0] == "negative"]
negative_tweets = random.sample(negative_tweets, len(positive_tweets))

neutral_tweets = [t for t in tweets if t["labels"][0] == "neutral"]
neutral_tweets = random.sample(neutral_tweets, len(positive_tweets))

tweets = positive_tweets + negative_tweets + neutral_tweets
random.shuffle(tweets)
n = len(tweets)
ratios = [int(train_ratio * n), int((train_ratio + dev_ratio) * n)]
train_tweets, val_tweets, test_tweets = np.split(tweets, ratios)

train_file = os.path.join(output_path, "train.json")
dev_file = os.path.join(output_path, "val.json")
test_file = os.path.join(output_path, "test.json")

with open(train_file, "w") as fh:
    json.dump(list(train_tweets), fh, indent=4)
with open(dev_file, "w") as fh:
    json.dump(list(val_tweets), fh, indent=4)
with open(test_file, "w") as fh:
    json.dump(list(test_tweets), fh, indent=4)