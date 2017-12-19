#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import os
from pprint import pprint

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np

yelp_file = 'data/yelp.csv'

if not os.path.isfile(yelp_file):
    print(yelp_file, ' is missing.')
    exit()

# 1. Loading dataset
yelp_df = pd.read_csv(yelp_file, sep=',', usecols=['stars', 'text'])

# 2. Feature matrix (X), response vector (y) and train_test_split
X = yelp_df.text
y = yelp_df.stars


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, shuffle=True)


# 3. Vectorize dataset
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)


# 2. Converting text and numbers (matrix)
# 3. training in ml (all can be written in one line)

models = {'Multinomial NaiveBayes': {'clf': MultinomialNB(),
                                     'clf_params': {
                                         'clf__alpha': (0.001, 1.0),
                                         'clf__fit_prior': (True, False),

                                     }},
          'SGDClassifier (SVM)': {'clf': SGDClassifier(loss='hinge', penalty='l2',
                                                       alpha=1e-3, random_state=1,
                                                       max_iter=5, tol=None),
                                  'clf_params': {
                                      'clf__alpha': (0.001, 1.0),
                                  }
                                  }}

for model in models.keys():
    print('\nRunning the model - {}'.format(model))
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', models[model]['clf'])])

    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)

    accuracy = np.mean(predicted == y_test)
    print('\nFirst run - Accuracy of {} - {}'.format(model, accuracy * 100))

    print('\nTuning training parameters')
    # 4. Auto-tuning the training parameters using Grid Search for both feature extraction and classifier
    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                  'vect__stop_words': ['english', None],
                  'vect__max_df': (0.5, 1.0),
                  'vect__min_df': (1, 2),
                  'tfidf__use_idf': (True, False),
                  'tfidf__smooth_idf': (True, False),
                  'tfidf__sublinear_tf': (True, False),
                  }

    parameters.update(models[model]['clf_params'])

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf.fit(X_train, y_train)

    gs_predicted = gs_clf.predict(X_test)
    accuracy = np.mean(gs_predicted == y_test)
    print('\nAfter tuning - Accuracy (after tuning) of {} - {}'.format(model, accuracy * 100))

    print('\nGrid Search best score -')
    print(gs_clf.best_score_)

    print('\nGrid Search best parameters -')
    pprint(gs_clf.best_params_)

    print('\nMetrics classification report ')
    print(metrics.classification_report(y_test, predicted))

    print('\nMetric Confusion matrix')
    print(metrics.confusion_matrix(y_test, predicted))


'''
Running the model - SGDClassifier (SVM)

First run - Accuracy of SGDClassifier (SVM) - 50.2

Tuning training parameters

After tuning - Accuracy (after tuning) of SGDClassifier (SVM) - 53.76

Grid Search best score -
0.5228

Grid Search best parameters -
{'clf__alpha': 0.001,
 'tfidf__smooth_idf': True,
 'tfidf__sublinear_tf': True,
 'tfidf__use_idf': True,
 'vect__max_df': 1.0,
 'vect__min_df': 1,
 'vect__ngram_range': (1, 3),
 'vect__stop_words': None}

Metrics classification report 
             precision    recall  f1-score   support

          1       0.52      0.47      0.49       191
          2       0.39      0.12      0.18       220
          3       0.45      0.15      0.23       379
          4       0.51      0.47      0.49       890
          5       0.51      0.81      0.62       820

avg / total       0.49      0.50      0.47      2500


Metric Confusion matrix
[[ 90  19   7  16  59]
 [ 47  26  39  57  51]
 [ 22   9  57 186 105]
 [  9  10  21 414 436]
 [  5   2   2 143 668]]

Running the model - Multinomial NaiveBayes

First run - Accuracy of Multinomial NaiveBayes - 42.92

Tuning training parameters

After tuning - Accuracy (after tuning) of Multinomial NaiveBayes - 51.2

Grid Search best score -
0.502533333333

Grid Search best parameters -
{'clf__alpha': 0.001,
 'clf__fit_prior': False,
 'tfidf__smooth_idf': True,
 'tfidf__sublinear_tf': False,
 'tfidf__use_idf': False,
 'vect__max_df': 1.0,
 'vect__min_df': 2,
 'vect__ngram_range': (1, 3),
 'vect__stop_words': None}

Metrics classification report 
             precision    recall  f1-score   support

          1       0.00      0.00      0.00       191
          2       0.00      0.00      0.00       220
          3       0.00      0.00      0.00       379
          4       0.39      0.93      0.55       890
          5       0.68      0.30      0.42       820

avg / total       0.36      0.43      0.33      2500


Metric Confusion matrix
[[  0   0   0 156  35]
 [  0   0   0 211   9]
 [  0   0   0 371   8]
 [  0   0   0 824  66]
 [  0   0   0 571 249]]
'''