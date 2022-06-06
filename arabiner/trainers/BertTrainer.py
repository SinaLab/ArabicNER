import os
import logging
import torch
import numpy as np
from arabiner.trainers import BaseTrainer
from arabiner.utils.metrics import compute_single_label_metrics

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

            for batch_index, (_, gold_tags, _, _, logits) in enumerate(self.tag(
                self.train_dataloader, is_train=True
            ), 1):
                self.current_timestep += 1
                batch_loss = self.loss(logits.view(-1, logits.shape[-1]), gold_tags.view(-1))
                batch_loss.backward()

                # Avoid exploding gradient by doing gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()
                self.scheduler.step()
                train_loss += batch_loss.item()

                if self.current_timestep % self.log_interval == 0:
                    logger.info(
                        "Epoch %d | Batch %d/%d | Timestep %d | LR %.10f | Loss %f",
                        epoch_index,
                        batch_index,
                        num_train_batch,
                        self.current_timestep,
                        self.optimizer.param_groups[0]['lr'],
                        batch_loss.item()
                    )

            train_loss /= num_train_batch

            logger.info("** Evaluating on validation dataset **")
            val_preds, segments, valid_len, val_loss = self.eval(self.val_dataloader)
            val_metrics = compute_single_label_metrics(segments)

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
                test_metrics = compute_single_label_metrics(segments)

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

    def eval(self, dataloader):
        golds, preds, segments, valid_lens = list(), list(), list(), list()
        loss = 0

        for _, gold_tags, tokens, valid_len, logits in self.tag(
            dataloader, is_train=False
        ):
            loss += self.loss(logits.view(-1, logits.shape[-1]), gold_tags.view(-1))
            preds += torch.argmax(logits, dim=2).detach().cpu().numpy().tolist()
            segments += tokens
            valid_lens += list(valid_len)

        loss /= len(dataloader)

        # Update segments, attach predicted tags to each token
        segments = self.to_segments(segments, preds, valid_lens, dataloader.dataset.vocab)

        return preds, segments, valid_lens, loss.item()

    def infer(self, dataloader):
        golds, preds, segments, valid_lens = list(), list(), list(), list()

        for _, gold_tags, tokens, valid_len, logits in self.tag(
            dataloader, is_train=False
        ):
            preds += torch.argmax(logits, dim=2).detach().cpu().numpy().tolist()
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
            pred = pred[1:valid_len-1]
            segment = segment[1:valid_len-1]

            for i, token in enumerate(segment):
                if vocab.tokens.stoi[token.text] != unk_id:
                    token.pred_tag = [{"tag": vocab.tags[0].itos[pred[i]]}]
                    tagged_segment.append(token)

            tagged_segments.append(tagged_segment)

        return tagged_segments
