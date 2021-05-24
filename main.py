# -----------------------------------------------------Imports-----------------------------------------------#

import numpy
import string
import pandas as pd
from joblib import dump
from sklearn import model_selection, preprocessing, decomposition, metrics, naive_bayes, linear_model, svm, ensemble, \
    pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from setup import setup

# -----------------------------------------------------Dataset Preparation-------------------------------------------------------#

numpy.random.seed(5312)
setup()
data = pd.read_csv('data/tamiltrain.csv', nrows=196)
# print(data.head())

# Tokenisation using indicNLP
# text_tokens = []
# for d in data['text']:
#     text_tokens.append(indic_tokenize.trivial_tokenize(d.strip('"')))
# # print(text_tokens)
# data['text_tokens'] = text_tokens
# print(data['text'].count())

# split the dataset into training and validation datasets
train_x, test_x, train_y, test_y = model_selection.train_test_split(data['text'], data['label'], test_size=0.25,
                                                                    random_state=5312)
print(train_x.shape, test_x.shape)

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)
# print(train_y, valid_y)

# ---------------------------------------------------------Feature Engineering-----------------------------------------#

# COUNT VECTORS
# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(data['text'])

# transform the training and validation data using count vectorizer object
xtrain_count = count_vect.transform(train_x)
xtest_count = count_vect.transform(test_x)
print(xtrain_count, "----End----", xtest_count)

# TF-IDF VECTORS
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(data['text'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xtest_tfidf = tfidf_vect.transform(test_x)
print(xtrain_tfidf, "\n----------END---------\n", xtest_tfidf)

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), max_features=5000)
tfidf_vect_ngram.fit(data['text'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xtest_tfidf_ngram = tfidf_vect_ngram.transform(test_x)
print(xtrain_tfidf_ngram, "\n----------END---------\n", xtest_tfidf_ngram)
#
# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)
tfidf_vect_ngram_chars.fit(data['text'])
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
xtest_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(test_x)
print(xtrain_tfidf_ngram_chars, "\n----------END---------\n", xtest_tfidf_ngram_chars)

# TEXT BASED FEATURES
data['char_count'] = data['text'].apply(len)
print(data['char_count'])
data['word_count'] = data['text'].apply(lambda x: len(x.split()))
print(data['word_count'])
data['word_density'] = data['char_count'] / (data['word_count'] + 1)
print(data['word_density'])
data['punctuation_count'] = data['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
print(data['punctuation_count'])


#
# # -------------------------------------------------- Training ML Models ---------------------------------------------- #
#
def train_model(classifier, feature_vector_train, label_train, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label_train)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    # predict the label on the traning data
    predict_train = classifier.predict(feature_vector_train)

    return metrics.accuracy_score(predictions, test_y)


# NAIVE BAYES CLASSIFIERS
# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xtest_count)
print("NB, Count Vectors: ", accuracy)

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xtest_tfidf)
print("NB, WordLevel TF-IDF: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram)
print("NB, N-Gram Vectors: ", accuracy)

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xtest_tfidf_ngram_chars)
print("NB, CharLevel Vectors: ", accuracy)

# LINEAR CLASSIFIER
# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xtest_count)
print("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xtest_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy)

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xtest_tfidf_ngram_chars)
print("LR, CharLevel Vectors: ", accuracy)

# SVM CLASSIFIER
# SVM on Count Vectors
accuracy = train_model(svm.SVC(), xtrain_count, train_y, xtest_count)
print("SVM, Count Vectors: ", accuracy)

# SVM on Word Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xtest_tfidf)
print("SVM, WordLevel TF-IDF: ", accuracy)

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram)
print("SVM, N-Gram Vectors: ", accuracy)

# SVM on Character Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y, xtest_tfidf_ngram_chars)
print("SVM, CharLevel Vectors: ", accuracy)

# RANDOM FOREST
# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xtest_count)
print("RF, Count Vectors: ", accuracy)

# RF on Word Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xtest_tfidf)
print("RF, WordLevel TF-IDF: ", accuracy)

# RF on Ngram Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram)
print("RF, N-Gram Vectors: ", accuracy)

# RF on Character Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram_chars, train_y, xtest_tfidf_ngram_chars)
print("RF, CharLevel Vectors: ", accuracy)

# # ______________________________________________ Pipeline ____________________________________________________________ #
pipeline = pipeline.Pipeline(steps=[('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)),
                                    ('model', ensemble.RandomForestClassifier())])
pipeline.fit(train_x, train_y)

# Dump the pipeline model
dump(pipeline, filename="tamil_sentence_classification.joblib")
