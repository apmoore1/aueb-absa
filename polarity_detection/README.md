# Aspect Based Sentiment Analysis

This is the software that accompanies the paper AUEB-ABSA at SemEval-2016 Task 5:
Supervised Machine Learning for Aspect Based Sentiment Analysis by Dionysios Xenos, 
Panagiotis Theodorakakos, John Pavlopoulos, Prodromos Malakasiotis and Ion Androutsopoulos. 

This paper describes our submissions to the Aspect Based Sentiment Analysis task of
SemEval-2016. For Aspect Category Detection (Subtask1/Slot1), we used multiple ensembles,
based on Support Vector Machine classifiers. For Opinion Target Expression extraction 
(Subtask1/Slot2), we used a sequence labeling approach with Conditional Random Fields.
For Polarity Detection (Subtask1/Slot3), we used an ensemble of two supervised classifiers,
one based on hand crafted features and one based on word embeddings. Our systems were ranked
in the top 6 positions in all the tasks we participated. The source code of our systems is 
publicly available.

## Getting started

```
Python version 2.7.10
Operating System : Linux(Ubuntu)
```

## Download

```
git clone https://github.com/nlpaueb/aueb-absa.git
cd aueb-absa/polarity_detection
```

If you are using virtualenv:

```
virtualenv env
source env/bin/activate
```

## Requirements

Install the required libraries using requirements.txt:

```
pip install -r requirements.txt 
```

or, by hand:

```
pip install numpy==1.11.0
pip install scipy==0.17.0
pip install scikit-learn==0.17.1
pip install nltk==3.2.1
```

## Extra

Download Punkt for NLTK:

```
python -c "import nltk;nltk.download('punkt')"
```

## Run

Unzip required files:

```
gunzip helpingEmbeddings.txt.gz
gunzip idf_amazon.txt.gz
```

Run the latest version:

```
python absa2016_v1.py
```

## Troubleshooting

Make sure that you have the following requirements installed:

```
sudo apt-get install build-essential
sudo apt-get install libreadline-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev gfortran libatlas-base-dev python-pip python-dev git
```

Make sure that you have a java compiler installed needed for the pos tagging:

```
sudo apt-get install openjdk-7-jdk
```
