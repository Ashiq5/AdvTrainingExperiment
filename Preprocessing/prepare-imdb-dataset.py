import os
import pandas as pd
import numpy as np
import re
import random
import math

movie_reviews = pd.read_csv("../dataset/IMDB Dataset.csv")
movie_reviews = movie_reviews.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Null: {movie_reviews.isnull().values.any()}")
print(f"shape: {movie_reviews.shape}")


TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    """
    Remove html tags
    """
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    """
    Remove html tags
    Remove punctuations and numbers
    Remove single character words
    Remove multiple spaces
    """
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


reviews = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    reviews.append(preprocess_text(sen))

for sen in sentences:
    print(sen)
    break

print(movie_reviews.columns.values)
movie_reviews.sentiment.unique()

y = movie_reviews['sentiment']
y = np.array(list(map(lambda x: 1 if x == "positive" else 0, y)))

print(sentences[0:5])
print(y[0:5])


import csv
def prepare_training_file():
    with open("../dataset/imdb-train.csv", mode='w') as out_file:
        csv_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['text', 'label'])
        for text, label in zip(sentences[0:15000], y[0:15000]):
            csv_writer.writerow([text, label])


def prepare_test_file():
    with open("../dataset/imdb-test.csv", mode='w') as out_file:
        csv_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['text', 'label'])
        for text, label in zip(sentences[15000:20000], y[15000:20000]):
            csv_writer.writerow([text, label])


prepare_training_file()
prepare_test_file()