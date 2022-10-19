# CL-teamlab - αNLI (Abductive Natural Language Inference)
Repository for the TeamLab project - Abductive NLI

Author: Esra Dönmez & Nianheng Wu

Our results ranked 3rd on leaderboard.

Full project report: [Here](https://github.com/esradonmez/CL-teamlab/blob/main/Final%20Report_Binary%20Classification%20vs.%20Ranking%20in%20Abductive%20Reasoning.pdf)

## Project Introduction:
The abductive natural language inference task (αNLI) is proposed to assess the abductive reasoning capability of machine  learning  systems. The task is to pick the hypothesis that is the most plausible explanation for a given observation pair. Details of this task could be find in [this paper](https://arxiv.org/abs/1908.05739).

In this repository, you can find our baseline model, which adopts BoW (Bag-of-Words) method with multilayer perceptron and defines the task as a simple binary classification problem. Beyond that, we propose that learn2rank framework might be more suitable than binary classification for this task. To test our hypothesis we conducted comparative analysis between these two approaches. We build our work on top of [this previous research](https://arxiv.org/pdf/2005.11223.pdf) and [their code](https://github.com/zycdev/L2R2) (In this paper, they adopted learn2rank framework for reasoning and achieved their best performance with RoBERTa). We extend their research and include a new pretrained model, [DeBERTa](https://arxiv.org/abs/2006.03654), which, according to [Allen AI leadboard](https://leaderboard.allenai.org/anli/submissions/public), should achieve better results than RoBERTa under binary classification framework. In this repository, we explore and compare the potential performance of these two models under learn2rank framework and binary classification.

This repository includes:
* Baseline approach - Multilayer perceptron with input initialized using GloVe embeddings
* Pretrained classifier - Binary classification model that inherits from pytorch lightning module
* learn2rank for reasoning approach - We only include the model classes (RoBERTa and DeBERTa) for this approach in the repository. Please refer to [this repository](https://github.com/RealNicolasBourbaki/L2R2) for the full code. (Please note that the code for this approach has been developed by https://github.com/zycdev/L2R2)

You can find more detailed descriptions under the content section.

## Data Preperation:
#### Download the data as:
```
wget https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip
unzip alphanli-train-dev.zip
```

#### Download the GloVe embeddings used in baseline as:
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove*.zip
```

## Content:
- ### CL-teamlab/baseline/
  - **baseline_anli.ipynb**
    - An independent and complete notebook file including all the code for the baseline model. File can be run on Google Colab, Jupyter notebook, etc..
  - **features.py**
    - Include ```Feature``` class that creates input embeddings from natural language input.
  - **perceptron.py**
    - Include ```MultiLayerPerceptron``` class.
  - **train.py**
    - See section "Run Models" to learn how to run this model.
  - **requirements.txt**
    - All dependencies that are needed for the baseline model. See section "Run Models" to learn how to install them.

- ### CL-teamlab/learn2rank/
  - **Credits**: https://github.com/zycdev/L2R2
  - Please refer to [this repository](https://github.com/RealNicolasBourbaki/L2R2) for the whole **updated code**. The **original code** is [here](https://github.com/zycdev/L2R2).
  - **../deberta/model.py**:
    - Include ```DebertaForListRank``` class, which initialize the model for DeBERTa learn2rank training.
  - **../roberta/model.py**:
    - Include ```RobertaForListRank``` class, which initialize the model for RoBERTa learn2rank training.

- ### CL-teamlab/pretrained_classifier/
  - **Credits**: https://github.com/isi-nlp/ai2
  - This folder contains the code for DeBERTa classifier and RoBERTa classifier. But the structure is the same and the code itself has little difference.
  - **config.yaml**
    - Configuration file for running the RoBERTa model
  - **config-deberta.yaml**
    - Configuration file for running the DeBERTa model
  - **environment.yml**
     - Environment settings
     - All dependencies are listed in this file.
  - **evaluate.py**
     - Run this file to get predictions.
     - If gold labels are available, it evaluates the predictions.
  - **model.py**
     - Include ```Classifier``` that inherits from pytorch lightning module.
     - Include dataloaders and collate function for abductive NLI.
     - If gold labels are available, it evaluates the predictions.
  - **train.py**
     - Run this file to train the model.

- ### utils
  - **\_\_init__.py**
  - **dataset.py**
    - Include ```Data``` class for the baseline model. The class represents αNLI dataset.
    - Include ```DataProcessor``` class for the baseline model. The class processes αNLI examples for the baseline MLP.
    - Include ```Anli``` class for pretrained model. The class represents αNLI dataset and inherits from Pytorch Dataset.
  - **evaluation.py**
    - Include ```Accuracy``` class to calculate simple binary accuracy between two lists.
  - **file_reader.py**
    - Include ```FileReader``` class to read tsv and jsonl files.
  - **preprocessor.py**
    - Include ```PrepareData``` class to preprocess natural language input. The class includes code for removing stopwords and punctuation, stemming, lemmatization, and removing unwanted characters.
  

## Run Models:
- ### Baseline model:
  - First, please download GloVe embeddings and αNLI dataset to any desired local repository. Learn how to download, please refer to the section **Data Preperation**.
  - Second, install all dependencies by running ```requirements.txt```:
    
    ```
    pip3 install -r requirements.txt
    ```
    
  - Then, please run the following command line on your terminal. Please substitute the content in ```[]``` to your customized values.
  
    ```
    python3 train.py 
        --train [path/to/the/training/set/] 
        --train_lbs [path/to/the/training/labels/] 
        --dev [path/to/the/dev/set/] 
        --dev_lbs [path/to/the/dev/labels/] 
        --test [path/to/the/test/set/] 
        --test_lbs [path/to/the/test/labels/] 
        --embed [path/to/the/embeddings/]
        --parameter_size [an integer, defining the size of parameters of the dense layers in the perceptron] 
        --batch_size [an interger, defining the training batch size] 
        --epochs [an integer, defining the maximum epochs]
    ```
  
- ### learn2rank models:
  - Please go to [this repository](https://github.com/RealNicolasBourbaki/L2R2) to run the whole **updated code**. Detailed instructions there.
  
- ### pretrained_classifier:
  - Make sure which model you would like to run (RoBERTa or DeBERTa), and go to the corresponding folder.
  - Create a virtual environment using conda and install all dependencies (We recommend creating separate environments for different models as they have different dependencies):
    ```
    conda env create -f environment.yml
    ```
  - Go to ```config.yaml``` (or if you are using DeBERTa, ```config-deberta.yaml```), and change the values for ```train_x```, ```train_y```, ```val_x```, ```val_y``` to the location where you stored the training & development set.
  - Run the code:
    ```
    python3 train.py
    ```
    
