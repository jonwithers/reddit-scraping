###
# This module contains cleaning and data visualization functions and classes used in the notebooks.
# The module is intended to avoid clutter in the notebooks and to provide scalability for the project.
######################################################################################################

###
# Imports
#########

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

###
# Functions
###########

def prepare_dataframe(raw_df, time, class_int):
    """Creates a time_on_reddit column, a total_text_column, and a class column. Drops unneccesary columns."""
    raw_df['time_on_reddit'] = time - raw_df['created_utc']
    raw_df.drop(['created_utc'], axis = 1, inplace=True)
    raw_df['total_text'] = raw_df['body'] + ' ' + raw_df['post_title']
    raw_df.drop(['body', 'post_title'], axis = 1, inplace=True)
    raw_df['length_of_text'] = raw_df['total_text'].map(lambda x: len(x))
    raw_df['class'] = class_int
    return raw_df

def top_20_words(corpus):
    """takes a corpus, returns a dataframe of the top 20 words and their counts"""
    cvec = CountVectorizer(stop_words='english')
    counts = cvec.fit_transform(corpus)
    return pd.DataFrame(([pd.DataFrame(counts.toarray(), columns = cvec.get_feature_names()).sum()])).T.sort_values(by = 0, ascending = False).head(20)

def make_loghist(col, xlab, ylab, outfile):
    """makes, labels, and saves a log histogram"""
    plt.figure(figsize=(8,8))
    sns.distplot(col, kde=False, hist_kws = {'log':True})
    plt.xlabel(xlab, fontsize = 18)
    plt.ylabel(ylab, fontsize = 18)
    plt.savefig('images/'+outfile)

def make_corrplot(x, y, df, xlab, ylab, outfile):
    """makes a corrplot with class shown by color"""

    mask0 = df['class']==0
    mask1 = df['class']==1

    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)
    ax.plot(x[mask0], y[mask0], '.', label='r/BPT', alpha = 0.3)
    ax.plot(x[mask1], y[mask1], '.', label='r/WPT', alpha = 0.3)
    plt.xlabel(xlab, fontsize = 18)
    plt.ylabel(ylab, fontsize = 18)
    fig.legend(fontsize = 18, markerscale = 3)
    plt.savefig("images/"+outfile)

def fit_and_make_report(model, X_train, y_train, X_test, y_test, show_best = False):
    """fits a model and makes a classification report, including how long it took to fit and predict."""
    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_elapsed = time.time() - fit_start

    if show_best:
        print(model.best_estimator_)

    score_start = time.time()
    cvs = cross_val_score(model, X_train, y_train, cv = 10)
    score_elapsed = time.time() - score_start
    print("Model fitting time:",fit_elapsed)
    print("Model scoring time:",score_elapsed)
    print("Mean cross-val score: {:.2%}".format(cvs.mean()))
    print("Cross-val score standard deviation: {:.2%}".format(cvs.std()))
    print("Model Score on test data: {:.2%}".format(model.score(X_test, y_test)))
    print("--")
    predict_start = time.time()
    predictions = model.predict(X_test)
    predict_elapsed = time.time() - predict_start
    print("Model prediction time:",predict_elapsed)
    print("Area under ROC curve Score on test data: {:.2%}".format(metrics.roc_auc_score(y_test, [i[1] for i in model.predict_proba(X_test)])))
    sns.heatmap(metrics.confusion_matrix(y_test, model.predict(X_test)), annot = True, fmt = 'g')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def show_best_features(fitted_model, outfile, transformer, numerics):
    """Takes a fitted model with a feature_importances_ attribute and provides visualization of the top features"""
    importance = fitted_model.feature_importances_
    cols = transformer.get_feature_names() + numerics

    plot_df = pd.DataFrame({
        'importance':importance,
        'feature':cols
    })


    most_important = plot_df.sort_values(by='importance', ascending = False).iloc[:30,:]
    most_important = most_important.sort_values(by = 'importance')
    most_important.set_index('feature', inplace=True)
    plt.figure(figsize=(8,8))
    ax = most_important.importance.plot(kind='barh',facecolor='#867899')
    plt.xlabel("Importance", fontsize = 18)
    plt.ylabel("Predictor", fontsize = 18)
    plt.savefig('images/'+outfile+'-Best-Predictors.png')

###
# Classes
#########

class RedditTransformer(BaseEstimator, TransformerMixin):
    """
    Prepares raw reddit data for modeling.
    """
    def __init__(self, class_int):
        self.class_int = class_int

    def transform(self, X):
        """
        transforms the text according to set steps.
        - prepare_dataframe
        -
        """
        X_copy = X.copy()
        X_copy = prepare_dataframe(X_copy, time=X['current_time'], class_int=self.class_int)
        X_copy.drop('current_time', axis = 1, inplace = True)

        numerics = ['comments_this_post',
                    'ups',
                    'time_on_reddit',
                    'length_of_text']
        target = 'class'
        minmax = MinMaxScaler()

        X_copy[numerics] = minmax.fit_transform(X_copy[numerics])

        return X_copy
    def fit_transform(self, X, y=None):
        self.fit()
        return self.transform(X)
