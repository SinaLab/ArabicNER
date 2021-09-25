ArabiNER
======================
Arabic language tagger for named entity recognition.


Requirements
--------
This package has dependencies on multiple Python packages. It is recommended to Conda to create a new environment 
that mimics the same environment the model was trained in. Provided in this repo `environment.yml` from which you 
can create a new conda environment using the command below.

    conda env create -f environment.yml

Inference
--------
Inference is the process of used a pre-trained model to perform tagging on a new text. To do that, we will 
need the following:

#### Model
The model can be downloaded [here](https://drive.google.com/file/d/1eWplRGCnxPkXGcczbYJJlytMG5tAWJaq/view?usp=sharing). 
Note that the model has the following structure and it is important to keep the same structure for inference to work.

    .
    ├── args.json
    ├── checkpoints
    │   ├── checkpoint_0.pt
    │   ├── checkpoint_1.pt
    │   ├── checkpoint_2.pt
    │   └── checkpoint_3.pt
    ├── predictions.txt
    ├── tag_vocab.pkl
    ├── tensorboard
    │   ├── Loss_test_loss
    │   │   └── events.out.tfevents.1632544966.ip-172-31-15-25.28258.3
    │   ├── Loss_train_loss
    │   │   └── events.out.tfevents.1632544966.ip-172-31-15-25.28258.1
    │   ├── Loss_val_loss
    │   │   └── events.out.tfevents.1632544966.ip-172-31-15-25.28258.2
    │   ├── Metrics_test_micro_f1
    │   │   └── events.out.tfevents.1632544966.ip-172-31-15-25.28258.7
    │   ├── Metrics_test_precision
    │   │   └── events.out.tfevents.1632544966.ip-172-31-15-25.28258.8
    │   ├── Metrics_test_recall
    │   │   └── events.out.tfevents.1632544966.ip-172-31-15-25.28258.9
    │   ├── Metrics_val_micro_f1
    │   │   └── events.out.tfevents.1632544966.ip-172-31-15-25.28258.4
    │   ├── Metrics_val_precision
    │   │   └── events.out.tfevents.1632544966.ip-172-31-15-25.28258.5
    │   ├── Metrics_val_recall
    │   │   └── events.out.tfevents.1632544966.ip-172-31-15-25.28258.6
    │   └── events.out.tfevents.1632544750.ip-172-31-15-25.28258.0
    └── train.log

#### Inference script
provided in the `bin` directory `infer.py` script that performs inference. 

The `infer.py` has the following parameters:

    usage: infer.py [-h] --output_path OUTPUT_PATH --model_path MODEL_PATH --text
                    TEXT [--batch_size BATCH_SIZE] [--max_seq_len MAX_SEQ_LEN]
                    [--overwrite]
    
    optional arguments:
      -h, --help            show this help message and exit
      --output_path OUTPUT_PATH
                            Output path to which logs are saved (default: None)
      --model_path MODEL_PATH
                            Model path for a pre-trained model, for this we you need to download the checkpoint from this repo  (default: None)
      --text TEXT           Text or sequence to tag, segments will be identified based on periods (default: None)
      --batch_size BATCH_SIZE
                            Batch size (default: 32)
      --max_seq_len MAX_SEQ_LEN
                            Maximum sequence length (default: 512)
      --overwrite           Overwrite output directory (default: False)

Example inference command:

    python -u /path/to/ArabiNER/arabiner/bin/infer.py 
           --output_path /path/to/some/dir 
           --model_path /path/to/model
           --overwrite 
           --text "وثائق نفوس شخصية من الفترة العثمانية للسيد نعمان عقل"

Features
--------



Credits
-------

