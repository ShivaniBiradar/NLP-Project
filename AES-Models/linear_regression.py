import data_utils
from qwk import quadratic_weighted_kappa

import tensorflow as tf
import os
import time

from sklearn import linear_model
from sklearn.model_selection import KFold
import numpy as np

# linear regression using 5-fold cross validation

## how to run this script:
## English data (essay sets 1-8)
## python linear_regression.py --essay_set_id <id>
##
## German data (essay sets 1, 2 or 10)
## python linear_regression.py --essay_set_id <id> --non_english_data


tf.flags.DEFINE_integer("essay_set_id", 1, "essay set id, 1 <= id <= 8")
tf.flags.DEFINE_boolean("non_english_data", False, "Use non-English dataset")

FLAGS = tf.flags.FLAGS
essay_set_id = FLAGS.essay_set_id
non_english = FLAGS.non_english_data

num_samples = 1
timestamp = time.strftime("%b_%d_%Y_%H-%M-%S", time.localtime())

# change output folder name based on language used
if (non_english):
    folder_name = '(DE) essay_set_{}_cv_{}_{}'.format(essay_set_id, num_samples, timestamp)
else:
    folder_name = '(EN) essay_set_{}_cv_{}_{}'.format(essay_set_id, num_samples, timestamp)

# different output folder for linear regression model
out_dir = os.path.abspath(os.path.join(os.path.curdir, "lr_runs", folder_name))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print("Writing to {}\n".format(out_dir))


# read dataset
if (non_english):
    # for the german data, the essay_set_id can be 1, 2 or 10
    training_file = "training_data/ASAP-DE/germanAsap.txt"
    essay_list, resolved_scores, _ = data_utils.load_german_training_data(training_file, essay_set_id)
else:
    training_path = "training_data/ASAP/training_set_rel3.tsv"
    essay_list, resolved_scores, _ = data_utils.load_training_data(training_path, essay_set_id)

print("Done reading essay data\n")

max_score = max(resolved_scores)
min_score = min(resolved_scores)

# only used for ASAP dataset
if essay_set_id == 7:
    min_score, max_score = 0, 30
elif essay_set_id == 8:
    min_score, max_score = 0, 60

# read word embeddings, but only indexes of words in the embedding list
if (non_english):
    # load German glove embeddings
    word_indexes, _ = data_utils.load_german_glove(index_only=True)
else:
    # load glove
    word_indexes, _ = data_utils.load_glove(42, dimensionality=300, index_only=True)

sent_size_list = list(map(len, [essay for essay in essay_list]))
max_sent_size = max(sent_size_list)

# return all essays in essay set as list of features for each essay
essay_data_list = data_utils.featurize_data(essay_list, word_indexes, max_sent_size)
fold_data = np.asarray(essay_data_list)


# linear regression features are:
# essay length (total number words), number of unique words, total number of sentences
# vector representation of essay is only used as a feature if the flag --use_essay is set

lm = linear_model.LinearRegression()
kf = KFold(n_splits=5)

kappa_scores = []

print("Running Linear Regression...\n")

for train_index, test_index in kf.split(fold_data):
    train_essays = []
    train_scores = []

    test_essays = []
    test_scores = []

    # add essays and scores to training set
    for i in train_index:
        train_essays.append(essay_data_list[i])
        train_scores.append(resolved_scores[i])

    # add essays and scores to testing set
    for i in test_index:
        test_essays.append(essay_data_list[i])
        test_scores.append(resolved_scores[i])

    lr_train_data = np.asarray(train_essays)
    lr_train_labels = np.asarray(train_scores)

    # fit the model
    model = lm.fit(lr_train_data, lr_train_labels)

    # make score predictions
    lr_predict_data = np.asarray(test_essays)

    predict_labels_np = lm.predict(lr_predict_data)
    predict_labels = []

    for pl in predict_labels_np:
        # keep predictions within the score range
        predict_labels.append(max(min(pl, max_score), min_score))

    # obtain QWK score for this split
    kappa_score = quadratic_weighted_kappa(test_scores, predict_labels, min_score, max_score)
    kappa_scores.append(kappa_score)

print("Writing QWK output")

with open(out_dir + '/eval.txt', 'a') as f:
    f.write('5 fold cv {}\n'.format(kappa_scores))
    f.write('final result is {}'.format(np.mean(np.array(kappa_scores))))

