Wojood - Nested Arabic NER
======================
Wojood is a corpus for Arabic nested Named Entity Recognition (NER). Nested entities occur 
when one entity mention is embedded inside another entity mention. 550K tokens (MSA and dialect)
This repo contains the source-code to train Wojood nested NER.

Online Demo
--------
You can try our model using the demo link below

https://ontology.birzeit.edu/Wojood/


Wojood Corpus
--------
A corpus and model for nested and flat Arabic Named Entity Recognition
Version: 1.0 (updated on 20/1/2022)

Wojood consists of about 550K tokens (MSA and dialect) that are manually 
annotated with 21 entity types (e.g., person, organization, location, event, date, etc). 
It covers multiple domains and was annotated with nested entities. The corpus contains 
about 75K entities and 22.5% of which are nested. A nested named entity recognition (NER)
model based on BERT was trained (F1-score 88.4%).

Corpus size: 550K tokens (MSA and dialects)

Richness: 21 entity classes, contains ~75K entities and 22.5% of them are nested entities

Domains: Media, History, Culture, Health, Finance, ICT, Law, Elections, Politics, Migration, Terrorism, social media

Inter-annotator agreement: 97.9% (Cohen's Kappa)

NER Model: AraBERTV2 (88.4% F1-score)

| Entity Classes (21):|||
|--------------------------------|------|------|
| PERS (person)                  |EVENT	 |   CARDINAL |
| NORP (group of people)	        | DATE	   | ORDINAL |
| OCC (occupation)	              | TIME	   | PERCENT |
| ORG (organization)	            | LANGUAGE|	QUANTITY |
| GPE (geopolitical entity)	     |  WEBSITE	|UNIT |
| LOC (geographical location)	   |     LAW	   | MONEY |
| FAC (facility: landmarks places) | PRODUCT	|CURR (currency) |

Please email Prof. Jarrar (mjarrar AT birzeit.edu) for the annotation guidelines


Corpus Download
--------
A sample data is available in the `data` directory. But the entire Wojood NER corpus is 
available to download upon request for academic and commercial use. Request to download 
Wojood (corpus and the model).

https://ontology.birzeit.edu/Wojood/

Model Download
--------
huggingface: https://huggingface.co/SinaLab/ArabicNER-Wojood

Requirements
--------
At this point, the code is compatible with `Python 3.10.6` and `torchtext==0.14.0`.

Clone this repo

    git clone https://github.com/SinaLab/ArabicNER.git

This package has dependencies on multiple Python packages. It is recommended to Conda to create a new environment 
that mimics the same environment the model was trained in. Provided in this repo `environment.yml` from which you 
can create a new conda environment using the command below.

    conda env create -f environment.yml

Update your PYTHONPATH to point to ArabicNER package

    export PYTHONPATH=PYTHONPATH:/path/to/ArabicNER

Model Training
--------
Argument for model traning are listed below. Note that some arguments including `data_config`, 
`trainer_config`, `network_config`, `optimizer`, `lr_scheduler` and `loss` take as input JSON 
configuration (see examples below).

    usage: train.py [-h] --output_path OUTPUT_PATH --train_path TRAIN_PATH
        --val_path VAL_PATH --test_path TEST_PATH
        [--bert_model BERT_MODEL] [--gpus GPUS [GPUS ...]]
        [--log_interval LOG_INTERVAL] [--batch_size BATCH_SIZE]
        [--num_workers NUM_WORKERS] [--data_config DATA_CONFIG]
        [--trainer_config TRAINER_CONFIG]
        [--network_config NETWORK_CONFIG] [--optimizer OPTIMIZER]
        [--lr_scheduler LR_SCHEDULER] [--loss LOSS] [--overwrite]
        [--seed SEED]
    
    optional arguments:
        -h, --help            show this help message and exit
        --output_path OUTPUT_PATH
            Output path (default: None)
        --train_path TRAIN_PATH
            Path to training data (default: None)
        --val_path VAL_PATH   
            Path to training data (default: None)
        --test_path TEST_PATH
            Path to training data (default: None)
        --bert_model BERT_MODEL
            BERT model (default: aubmindlab/bert-base-arabertv2)
        --gpus GPUS [GPUS ...]
            GPU IDs to train on (default: [0])
        --log_interval LOG_INTERVAL
            Log results every that many timesteps (default: 10)
        --batch_size BATCH_SIZE
            Batch size (default: 32)
        --num_workers NUM_WORKERS
            Dataloader number of workers (default: 0)
        --data_config DATA_CONFIG
            Dataset configurations (default: {"fn":
                "arabiner.data.datasets.DefaultDataset", "kwargs":
                {"max_seq_len": 512}})
        --trainer_config TRAINER_CONFIG
            Trainer configurations (default: {"fn":
            "arabiner.trainers.BertTrainer", "kwargs":
            {"max_epochs": 50}})
        --network_config NETWORK_CONFIG
            Network configurations (default: {"fn":
            "arabiner.nn.BertSeqTagger", "kwargs": {"dropout":
            0.1, "bert_model": "aubmindlab/bert-base-arabertv2"}})
        --optimizer OPTIMIZER
            Optimizer configurations (default: {"fn":
            "torch.optim.AdamW", "kwargs": {"lr": 0.0001}})
        --lr_scheduler LR_SCHEDULER
            Learning rate scheduler configurations (default:
                {"fn": "torch.optim.lr_scheduler.ExponentialLR",
                "kwargs": {"gamma": 1}})
        --loss LOSS           Loss function configurations (default: {"fn":
            "torch.nn.CrossEntropyLoss", "kwargs": {}})
        --overwrite           Overwrite output directory (default: False)
        --seed SEED           Seed for random initialization (default: 1)


#### Training nested NER example
In the case of nested NER we pass `NestedTagsDataset` to `--data_config`, `BertNestedTrainer` to `--trainer_config`,
and `BertNestedTagger` to `--network_config`.

    python train.py \
        --output_path /path/to/output/dir \
        --train_path /path/to/train.txt \
        --val_path /path/to/val.txt \
        --test_path /path/to/test.txt \
        --batch_size 8 \
        --data_config '{"fn":"arabiner.data.datasets.NestedTagsDataset","kwargs":{"max_seq_len":512}}' \
        --trainer_config '{"fn":"arabiner.trainers.BertNestedTrainer","kwargs":{"max_epochs":50}}' \
        --network_config '{"fn":"arabiner.nn.BertNestedTagger","kwargs":{"dropout":0.1,"bert_model":"aubmindlab/bert-base-arabertv2"}}' \
        --optimizer '{"fn":"torch.optim.AdamW","kwargs":{"lr":0.0001}}'

#### Training flat NER example
In the case of flat NER we pass `DefaultDataset` to `--data_config`, `BertTrainer` to `--trainer_config`,
and `BertSeqTagger` to `--network_config`.

    python train.py \
        --output_path /path/to/output/dir \
        --train_path /path/to/train.txt \
        --val_path /path/to/val.txt \
        --test_path /path/to/test.txt \
        --batch_size 8 \
        --data_config '{"fn":"arabiner.data.datasets.DefaultDataset","kwargs":{"max_seq_len":512}}' \
        --trainer_config '{"fn":"arabiner.trainers.BertTrainer","kwargs":{"max_epochs":50}}' \
        --network_config '{"fn":"arabiner.nn.BertSeqTagger","kwargs":{"dropout":0.1,"bert_model":"aubmindlab/bert-base-arabertv2"}}' \
        --optimizer '{"fn":"torch.optim.AdamW","kwargs":{"lr":0.0001}}'

Inference
--------
Inference is the process of using a pre-trained model to perform tagging on a new text. To do that, we will 
need the following:

#### Model
Note that the model has the following structure and it is important to keep the same structure for inference to work.

    .
    ├── args.json
    ├── checkpoints
    ├── predictions.txt
    ├── tag_vocab.pkl
    ├── tensorboard
    └── train.log

#### Inference script
provided in the `bin` directory `infer.py` script that performs inference. 

The `infer.py` has the following parameters:

    usage: infer.py [-h] --model_path MODEL_PATH --text
                    TEXT [--batch_size BATCH_SIZE] 
    
    optional arguments:
      -h, --help            show this help message and exit
      --model_path MODEL_PATH
                            Model path for a pre-trained model, for this we you need to download the checkpoint from this repo  (default: None)
      --text TEXT           Text or sequence to tag, segments will be identified based on periods (default: None)
      --batch_size BATCH_SIZE
                            Batch size (default: 32)
      
Example inference command:

    python -u /path/to/ArabiNER/arabiner/bin/infer.py
           --model_path /path/to/model
           --text "وثائق نفوس شخصية من الفترة العثمانية للسيد نعمان عقل"

#### Eval script
Optionally, there is `eval.py` script in `bin` directory to evaluate NER dataset with ground truth data.

    usage: eval.py [-h] --output_path OUTPUT_PATH --model_path MODEL_PATH
                    --data_paths DATA_PATHS [DATA_PATHS ...] [--batch_size BATCH_SIZE]
    
    optional arguments:
        -h, --help            show this help message and exit
        --output_path OUTPUT_PATH
            Path to save results (default: None)
        --model_path MODEL_PATH
            Model path (default: None)
        --data_paths DATA_PATHS [DATA_PATHS ...]
            Text or sequence to tag, this is in same format as
            training data with 'O' tag for all tokens (default: None)
        --batch_size BATCH_SIZE
            Batch size (default: 32)

Credits
-------
This research is partially funded by the Palestinian Higher Council for Innovation and Excellence.

Citation
-------
Mustafa Jarrar, Mohammed Khalilia, Sana Ghanem: [Wojood: Nested Arabic Named Entity Corpus and Recognition using BERT](http://www.jarrar.info/publications/JKG22.pdf ). In Proceedings of the International Conference on Language Resources and Evaluation (LREC 2022), Marseille, France. 2022

