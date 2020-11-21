import logging
import torch
import numpy as np

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
        log_interval=10,
    ):
        self.model = model
        self.max_epochs = max_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.log_interval = log_interval

    def train(self):
        current_timestep = 0
        best_val_loss, test_loss = np.inf, np.inf
        num_train_batch = len(self.train_dataloader)

        for epoch_index in range(self.max_epochs):
            for tweets, gold_labels, pred_labels in self.classify(
                self.train_dataloader
            ):
                current_timestep += 1
                train_loss = self.loss(pred_labels, gold_labels)
                train_loss.backward()
                self.optimizer.step()

                if current_timestep % self.log_interval == 0:
                    train_loss /= num_train_batch
                    logger.info(
                        "Epoch %d | Timestep %d | Train Loss %f | Val Loss %f | Test Loss %f",
                        epoch_index,
                        current_timestep,
                        train_loss,
                        best_val_loss,
                        test_loss,
                    )

            val_loss = self.eval(self.val_dataloader)
            logger.info(
                "Epoch %d | Timestep %d | Val Loss %f",
                epoch_index,
                current_timestep,
                val_loss,
            )

            self.scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info("Validation improved, evaluating test data...")
                test_loss = self.eval(self.test_dataloader)
                logger.info(
                    "Epoch %d | Timestep %d | Test Loss %f",
                    epoch_index,
                    current_timestep,
                    test_loss,
                )

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

            yield tweets, gold_labels, pred_labels

    def eval(self, dataloader):
        for tweets, gold_labels, pred_labels in self.classify(
            dataloader, is_train=False
        ):
            loss = self.loss(pred_labels, gold_labels)

        return loss / len(dataloader)
