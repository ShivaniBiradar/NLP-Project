import re
import os as os
import numpy as np
import itertools
import pandas as pd
from collections import Counter

## English ASAP dataset
def load_training_data(training_path, essay_set=1):
    training_df = pd.read_csv(training_path, delimiter='\t', encoding="ISO-8859-1")
    # resolved score for essay set 1
    resolved_score = training_df[training_df['essay_set'] == essay_set]['domain1_score']
    essay_ids = training_df[training_df['essay_set'] == essay_set]['essay_id']
    essays = training_df[training_df['essay_set'] == essay_set]['essay']
    essay_list = []

    # turn an essay to a list of words
    for idx, essay in essays.iteritems():
        essay = clean_str(essay)
        #essay_list.append([w for w in tokenize(essay) if is_ascii(w)])
        essay_list.append(tokenize(essay))

    return essay_list, resolved_score.tolist(), essay_ids.tolist()


## reading germanASAP.txt
def load_german_training_data(training_path, essay_set=1):
    ## pandas DataFrame for the training data
    ## encoding needs to be utf-8 in order to read non-English characters
    train_df = pd.read_csv(training_path, delimiter='\t', encoding="utf-8")

    ## make the score of the essay be score1 + score2
    score1 = train_df[train_df["EssaySet"] == essay_set]["Score1"]
    score2 = train_df[train_df["EssaySet"] == essay_set]["Score2"]
    resolved_score = score1 + score2

    ## obtain essay ids
    essay_ids = train_df[train_df["EssaySet"] == essay_set]["Id"]

    ## obtain text of essays
    essays = train_df[train_df["EssaySet"] == essay_set]["EssayText"]
    essay_list = []

    ## clean up and tokenize each essay
    for _, essay in essays.iteritems():
        essay = clean_german_str(essay)
        essay_list.append(tokenize(essay))

    return essay_list, resolved_score.tolist(), essay_ids.tolist()


def load_glove(token_num=6, dimensionality=50):
    word2vec = []
    word_indexes = {}
    # first word is nil
    word2vec.append([0] * dimensionality)
    count = 1

    ## open the glove file in the glove/ directory
    ## file contains word embeddings for token_num English words, embeddings are vector representations of words
    ## each embedding vector has dimensionality number of components

    ## glove file used is glove.42B.300d.txt, 42 billion tokens, 300 vector components each
    ## input parameters to this function are basically meaningless, only used for reading other glove files
    ## can modify this function to load a different set of word embeddings

    ## added this variable, original directory was called glove/
    word_embedding_path = "word_embeddings/"
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), word_embedding_path+"glove." + str(token_num) +
                           "B." + str(dimensionality) + "d.txt"), encoding="utf-8") as f:
        for line in f:
            l = line.split()
            word = l[0]
            vector = list(map(float, l[1:]))
            word_indexes[word] = count
            word2vec.append(vector)
            count += 1

    print("==> glove is loaded")

    ## return the indexes of all words in the glove file along with each word's vector representations
    return word_indexes, word2vec


## load German Glove embeddings, arguments are not needed since only one embedding file will be loaded
def load_german_glove():
    ## number of vector components per word
    dimensionality = 300

    word2vec = []
    word_indexes = {}
    word2vec.append([0] * dimensionality)
    count = 1

    word_embedding_path = "word_embeddings/german_glove.txt"

    ## do same operation as with English glove embeddings
    with open(word_embedding_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.split()
            word = line[0]
            vector = list(map(float, line[1:]))

            word_indexes[word] = count
            word2vec.append(vector)

            count += 1

    print("==> glove is loaded")

    ## return the indexes of all words in the glove file along with each word's vector representations
    return word_indexes, word2vec


def tokenize(sent):
    """
    Return the tokens of a sentence including punctuation.
    #>>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    #>>> tokenize('I don't know')
    ['I', 'don', '\'', 'know']
    """
    return [x.strip() for x in re.split('(\W+)', sent) if x.strip()]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

## new method for cleaning strings in german dataset
def clean_german_str(string):
    ## remove most special characters
    string = re.sub(r"[\"\'.\-\\/^:;%@#$&*_+=|<>~]", " ", string)

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

## this function is not actually used at all
def build_vocab(sentences, vocab_limit):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    print( 'Total size of vocab is {}'.format(len(word_counts.most_common())))
    # Mapping from index to word
    # vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_limit)]
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_limit)]
    
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i+1 for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

# data is DataFrame
def vectorize_data(data, word_indexes, sentence_size):
    vector_data = []
    
    for essay in data:
        length = max(0, sentence_size - len(essay))
        word_list = []

        for word in essay:
            if word in word_indexes:
                word_list.append(word_indexes[word])
            else:
                #print '{} is not in vocab'.format(w)
                word_list.append(0)

        word_list += [0] * length
        vector_data.append(word_list)

    return vector_data
