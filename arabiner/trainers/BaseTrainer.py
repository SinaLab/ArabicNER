import os
import torch
import logging

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
        output_path=None,
        vocab=None,
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
        self.vocab = vocab

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
