import logging
import torch
import numpy as np
from tweetclassifier.metrics import compute_metrics

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
            for tweets, gold_labels, logits in self.classify(
                self.train_dataloader, is_train=True
            ):
                current_timestep += 1
                train_loss = self.loss(logits, gold_labels)

                _, pred = torch.max(logits, dim=1)

                if current_timestep % self.log_interval == 0:
                    train_loss /= num_train_batch
                    logger.info(
                        "Epoch %d | Timestep %d | LR %f | Train Loss %f | Val Loss %f | Test Loss %f",
                        epoch_index,
                        current_timestep,
                        self.optimizer.param_groups[0]['lr'],
                        train_loss,
                        best_val_loss,
                        test_loss,
                    )

                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            val_loss, val_golds, val_preds = self.eval(self.val_dataloader)
            val_metrics = compute_metrics(val_golds.detach().cpu().numpy(), val_preds.detach().cpu().numpy())

            logger.info("Evaluating on validation dataset")
            logger.info(
                "Epoch %d | Timestep %d | Val Loss %f | F1 %f | Pr %f | Re %f | Acc %f",
                epoch_index,
                current_timestep,
                val_loss,
                val_metrics.f1,
                val_metrics.precision,
                val_metrics.recall,
                val_metrics.accuracy
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info("Validation improved, evaluating test data...")
                test_loss, test_golds, test_preds = self.eval(self.test_dataloader)
                test_metrics = compute_metrics(test_golds.detach().cpu().numpy(), test_preds.detach().cpu().numpy())

                logger.info(
                    "Epoch %d | Timestep %d | Test Loss %f | F1 %f | Pr %f | Re %f | Acc %f",
                    epoch_index,
                    current_timestep,
                    test_loss,
                    test_metrics.f1,
                    test_metrics.precision,
                    test_metrics.recall,
                    test_metrics.accuracy
                )

    def classify(self, dataloader, is_train=True):
        for tweets, gold_labels in dataloader:
            self.model.train(is_train)

            if torch.cuda.is_available():
                tweets = tweets.cuda()
                gold_labels = gold_labels.cuda()

            if is_train:
                self.optimizer.zero_grad()
                logits = self.model(tweets)
            else:
                with torch.no_grad():
                    logits = self.model(tweets)

            yield tweets, gold_labels, logits

    def eval(self, dataloader):
        golds = list()
        preds = list()

        for tweets, gold_labels, logits in self.classify(
            dataloader, is_train=False
        ):
            loss = self.loss(logits, gold_labels)
            golds.append(gold_labels)
            preds.append(logits)

        golds = torch.cat(golds, dim=0)
        preds = torch.argmax(torch.cat(preds, dim=0), dim=1)

        return loss / len(dataloader), golds, preds
