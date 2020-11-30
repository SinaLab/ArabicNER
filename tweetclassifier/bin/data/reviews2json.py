import csv
import json
import random
import numpy as np
import os

json_file = "/mnt/efs/data/amazon_product_data/automotive/reviews_Automotive_5.json"
output_path = "/mnt/efs/data/amazon_product_data/automotive"
reviews = list()
train_ratio = 0.7
dev_ratio = 0.1

with open(json_file, "r") as fh:
    for line in fh:
        j = json.loads(line)
        reviews.append({
            "tweet": j["summary"] + " " + j["reviewText"],
            "labels": [str(j["overall"])]
        })

random.shuffle(reviews)
n = len(reviews)
ratios = [int(train_ratio * n), int((train_ratio + dev_ratio) * n)]
train_reviews, val_reviews, test_reviews = np.split(reviews, ratios)

train_file = os.path.join(output_path, "train.json")
dev_file = os.path.join(output_path, "val.json")
test_file = os.path.join(output_path, "test.json")

with open(train_file, "w") as fh:
    json.dump(list(train_reviews), fh, indent=4)
with open(dev_file, "w") as fh:
    json.dump(list(val_reviews), fh, indent=4)
with open(test_file, "w") as fh:
    json.dump(list(test_reviews), fh, indent=4)