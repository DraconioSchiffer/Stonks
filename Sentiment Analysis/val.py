import time
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import MaxPooling1D, Embedding
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras.models import model_from_json

from keras.preprocessing.text import Tokenizer

from nltk.tokenize import word_tokenize

df1 = pd.read_csv("tweetdata.csv")
df1.columns = ['SentimentText', 'Sentiment']
df1 = df1[1::2]
corpus = df1['SentimentText']
labels = df1['Sentiment']

print('Corpus size: {}'.format(len(corpus)))




# WordVecs
def vec(corpus):
    vocab_size = 150000
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(corpus)

    return tokenizer



tokenized_corpus = vec(corpus)
vocab = 150000


# Creating sets
train_size = 2000
test_size = 659

wordIndices = np.zeros((2659, 25))
k = 0
for i in corpus:
    j = 0
    token = word_tokenize(i)
    for word in token:
        if(j < 25):
            try:
                wordIndices[k][j] = tokenized_corpus.word_index[word]
            except:
                wordIndices[k][j] = 0
        j = j + 1
    k = k+1

x = wordIndices
print(len(x))


x_train = x[:2000]
x_test = x[2000:]
max_tweet_length = 25
Y1 = df1[:train_size]
Y2 = df1[train_size:]
Y_train = pd.get_dummies(Y1['Sentiment']).values
Y_test = pd.get_dummies(Y2['Sentiment']).values



model = load_model('sentimentanalysis.model')
# Measuring Accuracy using Test Data
st_time = time.time()
score, acc = model.evaluate(x_test, Y_test, verbose=2, batch_size=20)
print("Logloss score: %.2f" % (score))
print("Validation set Accuracy: %.2f" % (acc))