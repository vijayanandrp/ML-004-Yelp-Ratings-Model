# ML-004-Yelp-Ratings-Model
------------------------------

### yelp_rating_predict_1.py

```text
--------------------------------------------------
 Accuracy of  MultinomialNB  -  0.909001956947

 Confusion matrix 
 [[119  76]
 [ 17 810]] 

 Area Under Curve of  MultinomialNB  -  0.941909279757 
--------------------------------------------------
--------------------------------------------------
 Accuracy of  LogisticRegression  -  0.93542074364

 Confusion matrix 
 [[147  48]
 [ 18 809]] 

 Area Under Curve of  LogisticRegression  -  0.969609028617 
--------------------------------------------------
 Finding top rating and non-rating words
 Total Features:  16800
 Total observations in each class  [  554.  2510.]
 -------------------- Top 20 five star rating words --------------------
             five_star  one_star  five_star_ratio
token                                            
fantastic          206         2        22.733865
perfect            235         4        12.967131
flavors            106         2        11.698008
outstanding         53         1        11.698008
yum                 52         1        11.477291
favorite           322         7        10.152988
ribs                45         1         9.932271
gluten              44         1         9.711554
mozzarella          42         1         9.270120
bianco              42         1         9.270120
gem                 38         1         8.387251
hubby               36         1         7.945817
pasty               35         1         7.725100
amazing            477        14         7.520148
delish              34         1         7.504382
organic             34         1         7.504382
waffles             34         1         7.504382
authentic           66         2         7.283665
superb              33         1         7.283665
die                 65         2         7.173307
 -------------------- Top 20 one star rating words --------------------
                five_star  one_star  five_star_ratio
token                                               
disgusting              1        30         0.007357
remove                  1        11         0.020065
unprofessional          1        10         0.022072
rude                    6        58         0.022833
pointing                1         9         0.024524
inedible                1         9         0.024524
flag                    1         9         0.024524
horrible                8        71         0.024870
refused                 2        17         0.025967
hubster                 1         8         0.027590
acknowledged            1         8         0.027590
ants                    1         8         0.027590
fedex                   1         8         0.027590
ugh                     2        16         0.027590
fuse                    1         8         0.027590
boca                    1         8         0.027590
unacceptable            1         8         0.027590
ignored                 2        15         0.029429
yuck                    2        15         0.029429
worst                   8        60         0.029429
```

### yelp_rating_predict_2.py

```text
Running the model - SGDClassifier (SVM)

First run - Accuracy of SGDClassifier (SVM) - 88.16046966731899

Tuning training parameters

After tuning - Accuracy (after tuning) of MultinomialNB (naive Bayes) - 90.21526418786692

Grid Search best score -
0.916449086162

Grid Search best parameters -
{'clf__alpha': 0.001,
 'tfidf__smooth_idf': True,
 'tfidf__sublinear_tf': True,
 'tfidf__use_idf': False,
 'vect__max_df': 0.5,
 'vect__min_df': 2,
 'vect__ngram_range': (1, 1),
 'vect__stop_words': 'english'}

Metrics classification report 
             precision    recall  f1-score   support

          0       0.96      0.39      0.56       195
          1       0.87      1.00      0.93       827

avg / total       0.89      0.88      0.86      1022


Metric Confusion matrix
[[ 77 118]
 [  3 824]]

Running the model - Multinomial NaiveBayes

First run - Accuracy of Multinomial NaiveBayes - 80.91976516634051

Tuning training parameters

After tuning - Accuracy (after tuning) of MultinomialNB (naive Bayes) - 91.78082191780823

Grid Search best score -
0.927545691906

Grid Search best parameters -
{'clf__alpha': 0.001,
 'clf__fit_prior': False,
 'tfidf__smooth_idf': True,
 'tfidf__sublinear_tf': False,
 'tfidf__use_idf': False,
 'vect__max_df': 1.0,
 'vect__min_df': 2,
 'vect__ngram_range': (1, 2),
 'vect__stop_words': None}

Metrics classification report 
/usr/local/lib/python3.5/dist-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
             precision    recall  f1-score   support

          0       0.00      0.00      0.00       195
          1       0.81      1.00      0.89       827

avg / total       0.65      0.81      0.72      1022


Metric Confusion matrix
[[  0 195]
 [  0 827]]

```
