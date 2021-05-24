import numpy
from joblib import dump
from sklearn import ensemble, preprocessing, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from setup import setup


# ----------------------------------------------------- Setup-----------------------------------
def train():
    numpy.random.seed(5312)
    setup()
    data = pd.read_csv('data/updated_dataset.csv')

    x = data['text']
    y = data['label']

    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(y)

    # ------------------------------------------ Pipeline ---------------------------------------------------------

    model = pipeline.Pipeline(
        steps=[('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)),
               ('model', ensemble.RandomForestClassifier())])
    model.fit(x, y)

    # Dump the pipeline model
    dump(model, filename="tamil_sentence_classification.joblib")
