# Detecting Target of Sarcasm Using Ensemble Methods
Code related to the ALTA's 2019 Shared Task - "Detecting Target of Sarcasm using Ensemble Methods". This was presented in ALTA's Conference. 

The link to the paper can be found -> http://bit.do/pradalta . Feel free to read the paper and to comment


# What's this Read Me about ? 
Basically , it is to give you an overview on what is this project about. Keep in mind that I've built this very quick and dirty way. It is not in the best shape but I am trying to make sure that at least people can download it and make it easier for them to download and use it 


## Requirements

* python 3.7
* numpy
* sckikit-learn
* pandas
* ALTA 2019 Shared Data Challenge Dataset (Please kindly obtain for them) 

Will update with pip requirements.txt so that you can have the exact one. I will make it easier for you all :). At the moment , I apologize - you just need to go through the files for the imports and install the pip To make your life a bit easier , the major imports or the stuff that I use can be found in the following files below :) 

## What are the files for

* DoItAll.py - It runs the rule-based system. Just load the CSV file with the title on the target. It will pick it up and get back to you
* LinearRegressionClassifer.py - The main linear regresson. You can take a look at the parameters that were used and also the way how we load the embeddings (.npy which was used from Google Sentence Encoder)
* lstm.py - This to generate the word embeddings - "Universal Sentence Encoder" . I should rename it , but this shows how you can quickly transform it
* rule_weighting.py - Our genetic algorithim in optimizing the weightages for it
* rules_implement.py - The rules from Joshi et al (2018) with our modifications. The original methods are still kept so that you may able to reproduce it 

## Problems Running ?

Raise an Issue or Reach out to me - I can help. Again it's my first time of publishing some public project. I try to follow best practices , so please pardon me if there are mistakes 

## TO-DO
* Improve the Read Me
* Refactor the Code - some of the file names do not even make sense or even the method. It can be optimized further 
* Add PIP Requirements
* Publish Jupyter-Notebook (to help people step-by-step and explain it to them) 
QUICK and DIRTY INFO


