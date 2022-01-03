import os
import logging
import torch
import numpy as np
from arabiner.trainers import BaseTrainer
from arabiner.utils.metrics import compute_nested_metrics

logger = logging.getLogger(__name__)


class BertCrfNestedTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        best_val_loss, test_loss = np.inf, np.inf
        num_train_batch = len(self.train_dataloader)

        for epoch_index in range(self.max_epochs):
            self.current_epoch = epoch_index
            train_loss = 0

            for batch_index, (_, gold_tags, _, _, batch_loss, _) in enumerate(self.tag(
                self.train_dataloader, is_train=True
            ), 1):
                self.current_timestep += 1
                torch.autograd.backward(batch_loss)

                # Avoid exploding gradient by doing gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()
                self.scheduler.step()
                bs = sum(l.item() for l in batch_loss)
                train_loss += bs

                if self.current_timestep % self.log_interval == 0:
                    logger.info(
                        "Epoch %d | Batch %d/%d | Timestep %d | LR %.10f | Loss %f",
                        epoch_index,
                        batch_index,
                        num_train_batch,
                        self.current_timestep,
                        self.optimizer.param_groups[0]['lr'],
                        bs
                    )

            train_loss /= num_train_batch

            logger.info("** Evaluating on validation dataset **")
            val_preds, segments, valid_len, val_loss = self.eval(self.val_dataloader)
            val_metrics = compute_nested_metrics(segments, self.val_dataloader.dataset.vocab.tags[1:])

            epoch_summary_loss = {
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            epoch_summary_metrics = {
                "val_micro_f1": val_metrics.micro_f1,
                "val_precision": val_metrics.precision,
                "val_recall": val_metrics.recall
            }

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
                logger.info("** Validation improved, evaluating test data **")
                test_preds, segments, valid_len, test_loss = self.eval(self.test_dataloader)
                self.segments_to_file(segments, os.path.join(self.output_path, "predictions.txt"))
                test_metrics = compute_nested_metrics(segments, self.test_dataloader.dataset.vocab.tags[1:])

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
            else:
                self.patience -= 1

            # No improvements, terminating early
            if self.patience == 0:
                logger.info("Early termination triggered")
                break

            self.summary_writer.add_scalars("Loss", epoch_summary_loss, global_step=self.current_timestep)
            self.summary_writer.add_scalars("Metrics", epoch_summary_metrics, global_step=self.current_timestep)

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
        for subwords, gold_tags, tokens, masks, valid_len in dataloader:
            self.model.train(is_train)

            if torch.cuda.is_available():
                subwords = subwords.cuda()
                gold_tags = gold_tags.cuda()

            if is_train:
                self.optimizer.zero_grad()
                loss = self.model(subwords, labels=gold_tags, masks=masks)
                pred_tags = None
            else:
                with torch.no_grad():
                    loss = self.model(subwords, labels=gold_tags, masks=masks)
                    pred_tags = self.model(subwords, masks=masks)

            yield subwords, gold_tags, tokens, valid_len, loss, pred_tags

    def eval(self, dataloader):
        golds, preds, segments, valid_lens = list(), list(), list(), list()
        loss = 0

        for _, gold_tags, tokens, valid_len, batch_loss, pred_tags in self.tag(
            dataloader, is_train=False
        ):
            loss += sum(l.item() for l in batch_loss)
            preds += pred_tags
            segments += tokens
            valid_lens += list(valid_len)

        loss /= len(dataloader)

        # Update segments, attach predicted tags to each token
        segments = self.to_segments(segments, preds, valid_lens, dataloader.dataset.vocab)

        return preds, segments, valid_lens, loss.item()

    def infer(self, dataloader):
        golds, preds, segments, valid_lens = list(), list(), list(), list()

        for _, gold_tags, tokens, valid_len, _, pred_tags in self.tag(
            dataloader, is_train=False
        ):
            preds += pred_tags
            segments += tokens
            valid_lens += list(valid_len)

        segments = self.to_segments(segments, preds, valid_lens, dataloader.dataset.vocab)
        return segments

    def to_segments(self, segments, preds, valid_lens, vocab):
        if vocab is None:
            vocab = self.vocab

        tagged_segments = list()
        unk_id = vocab.tokens.stoi["UNK"]

        for segment, pred, valid_len in zip(segments, preds, valid_lens):
            tagged_segment = list()
            # First, the token at 0th index [CLS] and token at nth index [SEP]
            pred = pred[1:valid_len-1, :]
            segment = segment[1:valid_len-1]

            for i, token in enumerate(segment):
                if vocab.tokens.stoi[token.text] != unk_id:
                    token.pred_tag = [{"tag": vocab.itos[tag_id]}
                                        for tag_id, vocab in zip(pred[i, :].int().tolist(), vocab.tags[1:])]
                    tagged_segment.append(token)

            tagged_segments.append(tagged_segment)

        return tagged_segments
