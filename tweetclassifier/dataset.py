import torch
import torchtext
from transformers import BertTokenizer


def get_dataloaders(input_path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    tweet_field = torchtext.data.Field(
        batch_first=True, tokenize=tokenizer.encode, use_vocab=False, pad_token=0
    )

    label_field = torchtext.data.LabelField(
        batch_first=True,
        use_vocab=True
    )

    fields = {"tweet": ("tweet", tweet_field), "label": ("label", label_field)}

    train_data, val_data, test_data = torchtext.data.TabularDataset.splits(
        path=input_path,
        train="train.json",
        validation="dev.json",
        test="test.json",
        fields=fields,
        format="json",
    )

    train_dataloader = torchtext.data.BucketIterator(
        train_data,
        batch_size=32,
        shuffle=True,
        train=True,
        device=torch.device("cuda:0"),
    )

    val_dataloader = torchtext.data.BucketIterator(
        val_data,
        batch_size=32,
        shuffle=True,
        train=True,
        device=torch.device("cuda:0")
    )

    test_dataloader = torchtext.data.BucketIterator(
        test_data,
        batch_size=32,
        shuffle=False,
        train=False,
        device=torch.device("cuda:0"),
    )

    label_field.build_vocab(train_data, val_data, test_data)

    return train_dataloader, val_dataloader, test_dataloader
