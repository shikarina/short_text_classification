#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:12:33 2021

@author: maksym.girnyk
"""
import pandas as pd
import re
import copy
import math
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from keras import backend as K
from collections import Counter

import itertools
from nltk.tokenize.treebank import TreebankWordDetokenizer

from statsmodels.stats.contingency_tables import mcnemar

# Extend the set of stopwords
stop_words = stopwords.words("english")
stop_words += ['rt', 'twitter', 'the', 'via', 'pic', 'com']


def lemmatize_words(text):
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    pos_tagged_text = nltk.pos_tag(text.split())
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

# Stratified sampling to have same proportion of 1s and 0s in train, valid and test sets
def split_train_valid_test_stratified(dataset, valid_share, test_share, label_col, seed):
    # Per-label subsets
    dataset1 = copy.deepcopy(dataset[dataset[label_col]==1])
    dataset0 = copy.deepcopy(dataset[dataset[label_col]==0])
    # Lengths of per-label test and validation subsets
    n_test1 = math.floor(test_share*len(dataset1))
    n_valid1 = math.floor(valid_share*len(dataset1))
    n_test0 = math.floor(test_share*len(dataset0))
    n_valid0 = math.floor(valid_share*len(dataset0))
    # Permute per-label subsets
    dataset1_perm = dataset1.sample(frac=1, random_state=seed)
    dataset0_perm = dataset0.sample(frac=1, random_state=seed)
    # (Stratified) sample training, validation and test sets
    dataset_test = pd.concat([dataset1_perm[0:n_test1], dataset0_perm[0:n_test0]], axis=0).sample(frac=1)
    dataset_valid = pd.concat([dataset1_perm[n_test1:n_test1+n_valid1], dataset0_perm[n_test0:n_test0+n_valid0]], axis=0).sample(frac=1)
    dataset_train = pd.concat([dataset1_perm[n_test1+n_valid1:], dataset0_perm[n_test0+n_valid0:]], axis=0).sample(frac=1)
    return dataset_train, dataset_valid, dataset_test


# Stratified sampling to have same proportion of 1s and 0s in both train and test sets
def split_train_test_stratified(dataset, test_share, label_col, seed):
    # Per-label subsets
    dataset1 = copy.deepcopy(dataset[dataset[label_col]==1])
    dataset0 = copy.deepcopy(dataset[dataset[label_col]==0])
    # Lengths of per-label test and validation subsets
    n_test1 = math.floor(test_share*len(dataset1))
    n_test0 = math.floor(test_share*len(dataset0))
    # Permute per-label subsets
    dataset1_perm = dataset1.sample(frac=1, random_state=seed)
    dataset0_perm = dataset0.sample(frac=1, random_state=seed)
    # (Stratified) sample training, validation and test sets
    dataset_test = pd.concat([dataset1_perm[0:n_test1], dataset0_perm[0:n_test0]], axis=0).sample(frac=1)
    dataset_train = pd.concat([dataset1_perm[n_test1:], dataset0_perm[n_test0:]], axis=0).sample(frac=1)
    return dataset_train, dataset_test


# Cleaning tweets
def preprocess_text(sentence):
    # Force UTF-8 encoding
    sen = sentence.force_encoding('utf-8').encode
    # Convert www.* or https?://* to URL
    sen = re.sub(r'http\S+', '', sen)
    sen = re.sub(r'pic.twitter.com/[\w]*',"", sen)
    # Convert to lower case
    sen = sen.lower()
    # Convert @username to AT_USER
    # sen = re.sub('@[^\s]+', 'USER', sen)
    sen = re.sub('@[^\s]+', '', sen)
    # Remove additional white spaces
    sen = re.sub('[\s]+', ' ', sen)
    # Replace #word with word
    sen = re.sub(r'#([^\s]+)', r'\1', sen)
    # Trim
    sen = sen.strip('\'"')
    # Remove punctuations and numbers
    sen = re.sub('[^a-zA-Z]', ' ', sen)
    # Single character removal
    sen = re.sub(r"\s+[a-zA-Z]\s+", ' ', sen)
    # Removing multiple spaces
    sen = re.sub(r'\s+', ' ', sen)
    # Split into tokens by white space
    tokens = sen.split()
    # Remove remaining tokens that are not alphabetic
    tokens = [token for token in tokens if token.isalpha()]
    # Filter out stop words
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatize_words(token) for token in tokens]
    # Filter out short tokens
    tokens = [token for token in tokens if len(token)>1]
    # Collect back to sentences
    sen = TreebankWordDetokenizer().detokenize(tokens)
    return sen


# Tokenize tweets
def tokenize_text(sentence):
    # Split into tokens by white space
    tokens = sentence.split()
    # Remove remaining tokens that are not alphabetic
    tokens = [token for token in tokens if token.isalpha()]
    # Filter out stop words
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize words:
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Filter out short tokens
    tokens = [token for token in tokens if len(token)>1]
    return tokens


# Compute performance metrics for a classifier
def compute_perf_metrics(labels_true, labels_pred):
    # Confusion matrix
    cnf_matrix = confusion_matrix(labels_true, labels_pred)
    tn, fp, fn, tp = cnf_matrix.ravel()
    # Accuracy
    acc = (tn+tp)/(tp+tn+fp+fn) 
    #Precision 
    prec = tp/(tp+fp) 
    # Recall 
    rec = tp/(tp+fn) 
    # F1 Score
    f1 = (2*prec*rec)/(prec + rec)
    return acc, prec, rec, f1


# Compute ROC curve (and AUC metric)
def compute_roc(labels_true, probs_pred):
    fpr, tpr, thresholds = roc_curve(labels_true, probs_pred)
    # roc_auc = auc(fpr, tpr)
    return fpr, tpr


# Compute PR curve (and AUC metric)
def compute_pr(labels_true, probs_pred):
    precision, recall, thresholds = precision_recall_curve(labels_true, probs_pred)
    # pr_auc = auc(recall, precision)
    return precision, recall


# Create embedding matrix for a neural network
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix


# Plot training history
def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    auc = history.history['auc_pr']
    val_auc = history.history['val_auc_pr']
    x = range(1, len(auc) + 1)

    plt.figure()
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Training and validation loss')
    plt.legend()

    plt.figure()
    plt.legend()
    plt.plot(x, auc, 'b', label='Training AUC')
    plt.plot(x, val_auc, 'r', label='Validation AUC')
    plt.ylabel('pr_auc')
    plt.xlabel('epoch')
    plt.title('Area under PR curve')
    plt.legend()
    

# Compute focal loss, as an alternative to binary cross-entropy loss for training
def focal_loss(y_true, y_pred):
    gamma = 2
    alpha = 0.1
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


# Compute metrics for training neural network in Keras
def recall(labels_true, labels_pred):
    true_positives = K.sum(K.round(K.clip(labels_true * labels_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(labels_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(labels_true, labels_pred):
    true_positives = K.sum(K.round(K.clip(labels_true * labels_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(labels_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(labels_true, labels_pred):
    prec = precision(labels_true, labels_pred)
    rec = recall(labels_true, labels_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))


# Plot confusion matrix    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute contingency table for pair-wise statistical testing 
def compute_contingency_table(pred_labels_1, pred_labels_2, test_labels):
    contingency_table = [[0, 0], [0, 0]]
    for label_idx in range(0, len(test_labels)):
        if (pred_labels_1[label_idx] == pred_labels_2[label_idx]):
            if (pred_labels_1[label_idx] == test_labels[label_idx]):
                contingency_table[1][1] += 1
            else:
                contingency_table[0][0] += 1
        else:
            if (pred_labels_1[label_idx] == test_labels[label_idx]):
                contingency_table[1][0] += 1
            elif (pred_labels_2[label_idx] == test_labels[label_idx]):
                contingency_table[0][1] += 1
            else:
                print("Something is wrong with label vectors!")
    print(contingency_table)
    return contingency_table

# Perform McNemar statistical tet
def perform_mcnemar_test(contigency_table, significance=0.05):
    test = mcnemar(contigency_table, exact=False, correction=True)
    if test.pvalue < significance:
        reject_null_hyp = 1
        # print("Reject Null Hypothesis")
        # print("Conclusion: Models have statistically different error rate")
    else:
        reject_null_hyp = 0
        # print("Accept Null Hypothesis")
        # print("Conclusion: Models do not have statistically different error rate")
    return test.statistic, test.pvalue, reject_null_hyp


# Plot statistical test results 
def plot_stats_matrix(mat, classes,
                          plot_type='CHI_2_STAT'):
    """
    Printing and plotting of the comparison matrix
    """
    plt.figure(figsize = (8,5))
    cmap = plt.get_cmap('Blues', 128)
    cmap.set_bad(color='darkred')
    plt.imshow(mat, interpolation='nearest', cmap=cmap)
   
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if plot_type=='CHI_2_STAT':
      title="McNemar test statistic"
    elif plot_type=='P_VAL':
      title="P-value"
    else:
      print("Wrong plot_type !!!");
    plt.title(title)
     
    threshold = np.nanmax(mat) / 2.
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        plt.text(j, i, round(mat[i, j], 2) if i!=j else "X",
                 horizontalalignment="center",
                 color="white" if mat[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel('Model 1')
    plt.xlabel('Model 2')


# Plot null hypothesis rejection matrix    
def plot_rejection_matrix(mat, classes):
    """
    Printing and plotting of the comparison matrix
    """
    plt.figure(figsize = (8,5))
    cmap = plt.get_cmap('Blues', 128)
    cmap.set_bad(color='darkred')
    plt.imshow(mat, interpolation='nearest', cmap=cmap)
   
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    title="Rejection of null hypothesis"
    plt.title(title)
    # cb = fig.colorbar() 
    # cb.remove() 
     
    threshold = np.nanmax(mat) / 2.
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        plt.text(j, i, ("Y" if mat[i, j]>=0.5 else "N") if i!=j else "X",
                 horizontalalignment="center",
                 color="white" if mat[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel('Model 1')
    plt.xlabel('Model 2')
    











