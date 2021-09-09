import os
import logging
import torch
import numpy as np
from arabiner.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class BaseTrainer:
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
        summary_writer=None,
        output_path=None
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
        self.summary_writer = summary_writer
        self.output_path = output_path
        self.current_timestep = 0
        self.current_epoch = 0

    def train(self):
        best_val_loss, test_loss = np.inf, np.inf
        num_train_batch = len(self.train_dataloader)

        for epoch_index in range(self.max_epochs):
            self.current_epoch = epoch_index
            train_loss = 0

            for batch_index, (_, gold_labels, logits) in enumerate(self.classify(
                self.train_dataloader, is_train=True
            ), 1):
                self.current_timestep += 1
                loss = self.loss(logits, gold_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                train_loss += loss.item()

                if self.current_timestep % self.log_interval == 0:
                    logger.info(
                        "Epoch %d | Batch %d/%d | Timestep %d | LR %f",
                        epoch_index,
                        batch_index,
                        num_train_batch,
                        self.current_timestep,
                        self.optimizer.param_groups[0]['lr'],
                    )

            train_loss /= num_train_batch
            val_loss, val_golds, val_preds = self.eval(self.val_dataloader)
            val_metrics = compute_metrics(val_golds.detach().cpu().numpy(), val_preds.detach().cpu().numpy())

            epoch_summary = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_metrics.f1,
                "val_precision": val_metrics.precision,
                "val_recall": val_metrics.recall,
                "val_accuracy": val_metrics.accuracy
            }

            logger.info("Evaluating on validation dataset")
            logger.info(
                "Epoch %d | Timestep %d | Train Loss %f | Val Loss %f | F1 %f | Pr %f | Re %f | Acc %f",
                epoch_index,
                self.current_timestep,
                train_loss,
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
                epoch_summary["test_loss"] = test_loss
                epoch_summary["test_f1"] = test_metrics.f1
                epoch_summary["test_precision"] = test_metrics.precision
                epoch_summary["test_recall"] = test_metrics.recall
                epoch_summary["test_accuracy"] = test_metrics.accuracy

                logger.info(
                    "Epoch %d | Timestep %d | Test Loss %f | F1 %f | Pr %f | Re %f | Acc %f",
                    epoch_index,
                    self.current_timestep,
                    test_loss,
                    test_metrics.f1,
                    test_metrics.precision,
                    test_metrics.recall,
                    test_metrics.accuracy
                )

                self.save()

            self.summary_writer.add_scalars("Loss", epoch_summary, global_step=self.current_timestep)

    def classify(self, dataloader, is_train=True):
        for text, gold_labels in dataloader:
            self.model.train(is_train)

            if torch.cuda.is_available():
                text = text.cuda()
                gold_labels = gold_labels.cuda()

            if is_train:
                self.optimizer.zero_grad()
                logits = self.model(text)
            else:
                with torch.no_grad():
                    logits = self.model(text)

            yield text, gold_labels, logits

    def eval(self, dataloader):
        golds = list()
        preds = list()

        for _, gold_labels, logits in self.classify(
            dataloader, is_train=False
        ):
            loss = self.loss(logits, gold_labels)
            golds.append(gold_labels)
            preds.append(logits)

        golds = torch.cat(golds, dim=0)
        preds = torch.argmax(torch.cat(preds, dim=0), dim=1)

        return loss.item() / len(dataloader), golds, preds

    def save(self):
        filename = os.path.join(
            self.output_path,
            "checkpoints",
            "checkpoint_{}.pt".format(self.current_epoch),
        )

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch
        }

        logger.info("Saving checkpoint to %s", filename)
        torch.save(checkpoint, filename)
