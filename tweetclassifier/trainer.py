import torch


class Trainer:
    def __init__(
        self,
        model,
        max_epochs=50,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
    ):
        self.model = model
        self.max_epochs = max_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def train(self):
        for i in range(self.max_epochs):
            for tweets, labels in self.train_dataloader:
                if torch.cuda.is_available():
                    tweets = tweets.cuda()
                    labels = labels.cuda()
