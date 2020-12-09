from itertools import chain
from tweetclassifier.dataset import TextTransform, parse_json

textcleaner = TextTransform()

data_files = (
    "/mnt/efs/data/smoke_tweets/train.json",
    "/mnt/efs/data/smoke_tweets/val.json",
    "/mnt/efs/data/smoke_tweets/test.json",
)

datasets, labels = parse_json(data_files)
dataset = list(chain(*datasets))


dataset = list(map(textcleaner, dataset))

print(len(dataset))