import torch
from torchvision import transforms
from transformers import BertModel
from itertools import chain
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from tweetclassifier.dataset import TextTransform, LabelTransform
from tweetclassifier.dataset import parse_json
from tweetclassifier.dataset import get_dataloaders

data_files = (
    #"/mnt/efs/data/smoke_tweets/01_All_Tweets.json",
    "/mnt/efs/data/smoke_tweets/train.json",
    "/mnt/efs/data/smoke_tweets/val.json",
    "/mnt/efs/data/smoke_tweets/test.json",
)

tsne = TSNE(n_components=2)
pca = PCA(n_components=2)

datasets, labels = parse_json(data_files)
dataset = list(chain(*datasets))
bert = BertModel.from_pretrained("bert-base-uncased").eval().cuda()
transform = transforms.Compose(
    [TextTransform("bert-base-uncased"), LabelTransform(labels)]
)

dataloader = get_dataloaders(
    (dataset,), transform=transform, batch_size=256, shuffle=(False,)
)[0]

Y, X = list(), list()

with torch.no_grad():
    for batch_index, (text, labels) in enumerate(dataloader):
        print(batch_index, len(dataloader))
        text = text.cuda()
        _, x = bert(text)

        Y.append(labels)
        X.append(x)

    Y = torch.cat(Y, dim=0)
    X = torch.cat(X, dim=0)

Y = Y.cpu().numpy()
X = X.cpu().numpy()
#X2d = pca.fit_transform(X)
X2d = tsne.fit_transform(X)
print(X2d.shape)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

plt.scatter(X2d[:, 0], X2d[:, 1], c=Y, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()
