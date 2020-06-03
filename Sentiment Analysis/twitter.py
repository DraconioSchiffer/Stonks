import pandas as pd 
import numpy as np 
from copy import deepcopy
from string import punctuation
from random import shuffle
import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
import re as re
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import MaxPooling1D, Embedding
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils import to_categorical
import time




pd.options.mode.chained_assignment = None
tokenizer = TweetTokenizer()
tqdm.pandas(desc="progress-bar")
LabeledSentence = gensim.models.doc2vec.LabeledSentence
n = 2569
n_dim=200
vocab = 150000

# Import Data
data = pd.read_csv("tweetdata.csv")
data.columns = ['SentimentText', 'Sentiment']
data = data[1::2]
data['SentimentText'] = data[data['SentimentText'].isnull() == False]


def tokenize(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) 
    tokens = tokenizer.tokenize(tweet)
    return tokens


def postprocess(data, n=2659):
    data = data.head(n)
    data['tokens'] = data['SentimentText'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    #data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(data)
print(data)

#Dividing into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens), np.array(data.head(n).Sentiment), test_size=0.25)

#labeling tweets
def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

print(x_train[0])
x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')
print(x_train[0])

#training Word2Vecs
tweet_w2v = Word2Vec(size=n_dim, min_count=2)
xw = []#list(x.words for x in tqdm(x_train))
for x in tqdm(x_train):
    xw.append(x.words)
tweet_w2v.build_vocab(xw)
tweet_w2v.train(xw, total_examples = len(xw), epochs = 10)

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print ('vocab size :', len(tfidf))

#build complete vector tweets
def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
print(train_vecs_w2v)
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)
print("HERE SAFELY")




# CNN
batch_size = 20
nb_epochs = 10

model = Sequential()
model.add(Embedding(vocab, 64, input_length=200))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])
print(model.summary())

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Fit the model
model.fit(train_vecs_w2v, y_train,
          batch_size=batch_size,
          shuffle=True,
          epochs=5,
          validation_split=0.2,
          verbose=2)

# Measuring Accuracy using Test Data
st_time = time.time()
score, acc = model.evaluate(test_vecs_w2v, y_test, verbose=2, batch_size=20)
print("Logloss score: %.2f" % (score))
print("Validation set Accuracy: %.2f" % (acc))

# Save Model
model.save('sentimentanalysis.model')

# Serialize model to json
multi = model.to_json()
with open("tweets.json", "w") as json_file:
    json_file.write(multi)

## Serialize weights to HDF5
model.save_weights("tweets.h5")
print("Model saved successfully!")