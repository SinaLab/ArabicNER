import os
import torch
import logging
import natsort
import glob

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        model=None,
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
        clip=5,
        patience=5
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
        self.clip = clip
        self.patience = patience

    def tag(self, dataloader, is_train=True):
        """
        Given a dataloader containing segments, predict the tags
        :param dataloader: torch.utils.data.DataLoader
        :param is_train: boolean - True for training model, False for evaluation
        :return: Iterator
                    subwords (B x T x NUM_LABELS)- torch.Tensor - BERT subword ID
                    gold_tags (B x T x NUM_LABELS) - torch.Tensor - ground truth tags IDs
                    tokens - List[arabiner.data.dataset.Token] - list of tokens
                    valid_len (B x 1) - int - valiud length of each sequence
                    logits (B x T x NUM_LABELS) - logits for each token and each tag
        """
        for subwords, gold_tags, tokens, valid_len in dataloader:
            self.model.train(is_train)

            if torch.cuda.is_available():
                subwords = subwords.cuda()
                gold_tags = gold_tags.cuda()

            if is_train:
                self.optimizer.zero_grad()
                logits = self.model(subwords)
            else:
                with torch.no_grad():
                    logits = self.model(subwords)

            yield subwords, gold_tags, tokens, valid_len, logits

    def segments_to_file(self, segments, filename):
        """
        Write segments to file
        :param segments: [List[arabiner.data.dataset.Token]] - list of list of tokens
        :param filename: str - output filename
        :return: None
        """
        with open(filename, "w") as fh:
            results = "\n\n".join(["\n".join([t.__str__() for t in segment]) for segment in segments])
            fh.write("Token\tGold Tag\tPredicted Tag\n")
            fh.write(results)
            logging.info("Predictions written to %s", filename)

    def save(self):
        """
        Save model checkpoint
        :return:
        """
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

    def load(self, checkpoint_path):
        """
        Load model checkpoint
        :param checkpoint_path: str - path/to/checkpoints
        :return: None
        """
        checkpoint_path = natsort.natsorted(glob.glob(f"{checkpoint_path}/checkpoint_*.pt"))
        checkpoint_path = checkpoint_path[-1]

        logger.info("Loading checkpoint %s", checkpoint_path)

        device = None if torch.cuda.is_available() else torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model"])
