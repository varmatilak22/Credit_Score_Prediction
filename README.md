
# Spam SMS Classification Project 📱🚫
## Overview
This project focuses on classifying SMS messages as spam or non-spam (ham) using natural language processing (NLP) techniques. The goal is to develop machine learning models based on decision tree and random forest algorithms to automatically identify spam content in SMS messages.

Project link: [https://colab.research.google.com/github/tilakrvarma22/installer/blob/main/Spam_SMS_Classification.ipynb](https://colab.research.google.com/drive/1BfWT9b20XdI0Mpvz3Cib9KM8WQiQiveo)

## Table of Contents
* Getting Started
* Prerequisites
* Installation
* Usage
* Data
* NLP Techniques
* Model Training
* Evaluation
* Results
* Contributing
* License
* Acknowledgments

## Getting Started
These instructions will help you set up the project on your local machine for development and testing purposes.

## Prerequisites
Make sure you have Python installed along with the necessary libraries. You can download Python from python.org and install libraries using pip.

## Data
The dataset used for this project is available in the data directory. The main file is spam_sms_data.csv, containing text data labeled as spam or non-spam (ham).

## NLP Techniques
NLP techniques such as tokenization, stemming, and TF-IDF vectorization are utilized to preprocess the text data before model training. These techniques help in transforming raw text into numerical features that can be used by machine learning algorithms.

1. Bag of Words (BoW):

BoW creates a list of all unique words in a text and counts how many times each word appears. It represents text as a collection of word counts.

2. Tokenization:

Tokenization splits text into smaller units called tokens, which can be words, phrases, or other meaningful elements.

3. Stemming:

Stemming reduces words to their root form by removing suffixes and prefixes.

4. TF-IDF (Term Frequency-Inverse Document Frequency):

TF-IDF measures the importance of a word in a document relative to a collection of documents. It consists of two parts:
Term Frequency (TF): Measures how often a word appears in a document.
Inverse Document Frequency (IDF): Measures how unique a word is across all documents in the collection.

5. Count Vectorizer:

Count Vectorizer converts text documents into numerical representations by counting the occurrences of words in each document.
6. TfidfVectorizer:

TfidfVectorizer combines the TF-IDF weighting scheme with Count Vectorizer to produce a matrix of TF-IDF features.

## Model Training
The model training process is documented in the train_and_evaluate.py script. Both decision tree and random forest algorithms are used to train models on the preprocessed text data.

1. Decision Tree:
Decision trees are a popular machine learning model for classification tasks. They make predictions by recursively partitioning the feature space based on feature values.
Each node in a decision tree represents a decision based on a feature, leading to one of the possible outcomes (classes).
2. Random Forest:
Random forests are an ensemble learning method that combines multiple decision trees to make predictions. Each tree in the forest is trained independently on a random subset of the training data and features.
* Ensemble Techniques
1. Bagging (Bootstrap Aggregating):
Bagging involves training multiple models independently on different subsets of the training data, sampled with replacement (bootstrap samples).
2. Boosting:
Boosting combines multiple weak learners to create a strong learner. It trains models sequentially, where each new model focuses on examples that the previous models misclassified.

## Evaluation
The performance of the trained models is evaluated using metrics such as accuracy, precision, recall, and F1-score. Detailed information is available in the evaluation section of the train_and_evaluate.py script.

## Results
The final trained models are saved in the models directory. You can use them to make predictions on new SMS messages.

## Contributing
If you'd like to contribute to this project, please open an issue or create a pull request. All contributions are welcome! 🙌

## License
This project is licensed under the GECCS License - see the LICENSE file for details.
