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
yelp_df = yelp_df.loc[(yelp_df.stars == 5) | (yelp_df.stars == 1), :]
yelp_df.stars_map = yelp_df.stars.map({5: 1, 1: 0})

# 2. Feature matrix (X), response vector (y) and train_test_split
X = yelp_df.text
y = yelp_df.stars_map


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
'''