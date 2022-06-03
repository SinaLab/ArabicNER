Wojood Arabic NER
======================
Wojood is a corpus for Arabic nested Named Entity Recognition (NER). Nested entities occur 
when one entity mention is embedded inside another entity mention. Wojood consists 
of about 550K Modern Standard Arabic (MSA) and dialect tokens that are manually 
annotated with 21 entity types including person, organization, location, event 
and date. More importantly, the corpus is annotated with nested entities instead 
of the more common flat annotations. The data contains about 75K entities and 22.5% of 
which are nested. The inter-annotator evaluation of the corpus demonstrated a strong 
agreement with Cohen's Kappa of 0.979 and an F1-score of 0.976. To validate our data, 
we used the corpus to train a nested NER model based on multi-task learning 
and AraBERT (Arabic BERT). This repo contains the source-code to train Wojood nested NER.

Wojood Corpus
--------
A corpus and model for nested Arabic Named Entity Recognition
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

Online Demo
--------
You can try our model using the demo link below

https://ontology.birzeit.edu/Wojood/

Corpus Download
--------
A sample data is available in the `data` directory. But the entire Wojood NER corpus is 
available to download upon request for academic and commercial use. Request to download 
Wojood (corpus and the model).

https://ontology.birzeit.edu/Wojood/

Requirements
--------
Clone this repo

    git clone https://github.com/SinaLab/ArabicNER.git

This package has dependencies on multiple Python packages. It is recommended to Conda to create a new environment 
that mimics the same environment the model was trained in. Provided in this repo `environment.yml` from which you 
can create a new conda environment using the command below.

    conda env create -f environment.yml

Update your PYTHONPATH to point to ArabiNER package

    export PYTHONPATH=PYTHONPATH:/path/to/ArabiNER

Inference
--------
Inference is the process of used a pre-trained model to perform tagging on a new text. To do that, we will 
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

Credits
-------
This research is partially funded by the Palestinian Higher Council for Innovation and Excellence.

Citation
-------
Mustafa Jarrar, Mohammed Khalilia, Sana Ghanem: [Wojood: Nested Arabic Named Entity Corpus and Recognition using BERT](http://www.jarrar.info/publications/JKG22.pdf ). In Proceedings of the International Conference on Language Resources and Evaluation (LREC 2022), Marseille, France. 2022

