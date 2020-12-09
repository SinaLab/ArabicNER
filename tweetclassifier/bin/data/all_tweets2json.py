import os
import csv
import json
import random
import numpy as np

json_file = "/mnt/efs/data/smoke_tweets/01_All_Tweets.csv"
output_path = "/mnt/efs/data/smoke_tweets/sentiment"
tweets = list()
train_ratio = 0.7
dev_ratio = 0.1

with open(json_file, "r", encoding="utf-8") as fh:
    rows = csv.reader(fh, delimiter=",")
    next(rows)
    for row in rows:
        tweets.append({
            "tweet": row[9],
            "labels": [row[17]]
        })

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
