import logging
import torch

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        max_epochs=50,
        optimizer=None,
        scheduler=None,
        loss=None,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
    ):
        self.model = model
        self.max_epochs = max_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss

    def train(self):
        for i in range(self.max_epochs):
            for tweets, gold_labels, pred_labels in self.classify(
                self.train_dataloader
            ):
                loss = self.loss(gold_labels, pred_labels)
                loss.backward()
                self.optimizer.step()
                logger.info(loss.item())

            self.scheduler.step()

    def classify(self, dataloader, is_train=True):
        for tweets, gold_labels in dataloader:
            self.model.train(is_train)

            if torch.cuda.is_available():
                tweets = tweets.cuda()
                gold_labels = gold_labels.cuda()

            if is_train:
                self.optimizer.zero_grad()
                pred_labels = self.model(tweets)
            else:
                with torch.no_grad():
                    pred_labels = self.model(tweets)

            yield tweets, gold_labels, pred_labels[0]
