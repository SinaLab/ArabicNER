import logging
import torch
import numpy as np
from arabiner.data.dataset import Token
from arabiner.trainers import BaseTrainer
from arabiner.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class BertTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        best_val_loss, test_loss = np.inf, np.inf
        num_train_batch = len(self.train_dataloader)

        for epoch_index in range(self.max_epochs):
            self.current_epoch = epoch_index
            train_loss = 0

            for batch_index, (_, gold_tags, _, logits, _) in enumerate(self.tag(
                self.train_dataloader, is_train=True
            ), 1):
                self.current_timestep += 1
                batch_loss = self.loss(logits.view(-1, logits.shape[-1]), gold_tags.view(-1))
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                train_loss += batch_loss.item()

                if self.current_timestep % self.log_interval == 0:
                    logger.info(
                        "Epoch %d | Batch %d/%d | Timestep %d | LR %f | Loss %f",
                        epoch_index,
                        batch_index,
                        num_train_batch,
                        self.current_timestep,
                        self.optimizer.param_groups[0]['lr'],
                        batch_loss.item()
                    )

            train_loss /= num_train_batch
            val_loss, val_golds, val_preds, tokens, valid_len = self.eval(self.val_dataloader)
            segments = self.to_segments(val_golds, val_preds, tokens, valid_len)
            val_metrics = compute_metrics(segments)

            epoch_summary_loss = {
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            epoch_summary_metrics ={
                "val_micro_f1": val_metrics.micro_f1,
                "val_precision": val_metrics.precision,
                "val_recall": val_metrics.recall
            }

            logger.info("Evaluating on validation dataset")
            logger.info(
                "Epoch %d | Timestep %d | Train Loss %f | Val Loss %f | F1 %f",
                epoch_index,
                self.current_timestep,
                train_loss,
                val_loss,
                val_metrics.micro_f1
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info("Validation improved, evaluating test data...")
                test_loss, test_golds, test_preds, tokens, valid_len = self.eval(self.test_dataloader)
                segments = self.to_segments(test_golds, test_preds, tokens, valid_len)
                self.segments_to_file(segments)
                test_metrics = compute_metrics(segments)
                epoch_summary_loss["test_loss"] = test_loss
                epoch_summary_metrics["test_micro_f1"] = test_metrics.micro_f1
                epoch_summary_metrics["test_precision"] = test_metrics.precision
                epoch_summary_metrics["test_recall"] = test_metrics.recall

                logger.info(
                    f"Epoch %d | Timestep %d | Test Loss %f | F1 %f",
                    epoch_index,
                    self.current_timestep,
                    test_loss,
                    test_metrics.micro_f1
                )

                self.save()

            self.summary_writer.add_scalars("Loss", epoch_summary_loss, global_step=self.current_timestep)
            self.summary_writer.add_scalars("Metrics", epoch_summary_metrics, global_step=self.current_timestep)

    def tag(self, dataloader, is_train=True):
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

            yield subwords, gold_tags, tokens, logits, valid_len

    def eval(self, dataloader):
        golds, preds, tokens, valid_lens = list(), list(), list(), list()
        loss = 0

        for _, gold_tags, batch_tokens, logits, valid_len in self.tag(
            dataloader, is_train=False
        ):
            loss += self.loss(logits.view(-1, logits.shape[-1]), gold_tags.view(-1))
            golds += gold_tags.detach().cpu().numpy().tolist()
            preds += torch.argmax(logits, dim=2).detach().cpu().numpy().tolist()
            tokens += batch_tokens.detach().cpu().numpy().tolist()
            valid_lens += list(valid_len)

        loss /= len(dataloader)

        return loss.item(), golds, preds, tokens, valid_lens

    def infer(self, dataloader, vocab):
        golds, preds, tokens, valid_lens = list(), list(), list(), list()

        for _, gold_tags, batch_tokens, logits, valid_len in self.tag(
            dataloader, is_train=False
        ):
            golds += gold_tags.detach().cpu().numpy().tolist()
            preds += torch.argmax(logits, dim=2).detach().cpu().numpy().tolist()
            tokens += batch_tokens.detach().cpu().numpy().tolist()
            valid_lens += list(valid_len)

        segments = self.to_segments(golds, preds, tokens, valid_lens, vocab=vocab)
        return segments

    def to_segments(self, golds, preds, text, valid_lens, vocab=None):
        if vocab is None:
            vocab = self.vocab

        segments = list()
        unk_id = vocab.tokens.stoi["UNK"]

        for gold, pred, tokens, valid_len in zip(golds, preds, text, valid_lens):
            # First, the token at 0th index [CLS] and token at nth index [SEP]
            gold = gold[1:valid_len-1]
            pred = pred[1:valid_len-1]
            tokens = tokens[1:valid_len-1]

            segment = [Token(
                text=vocab.tokens.itos[t],
                pred_tag=vocab.tags.itos[p],
                gold_tag=vocab.tags.itos[g])
                for t, p, g in zip(tokens, pred, gold) if t != unk_id]

            segments.append(segment)

        return segments
