from keras.models import load_model
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
import time




pd.options.mode.chained_assignment = None
tokenizer = TweetTokenizer()
tqdm.pandas(desc="progress-bar")
LabeledSentence = gensim.models.doc2vec.LabeledSentence
n_dim=200
vocab = 150000

# Import Data
data = pd.read_csv("recent_tweets.csv")

n = len(data)



def tokenize(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) 
    tokens = tokenizer.tokenize(tweet)
    return tokens


def postprocess(data, n):
    data = data.head(n)
    data['tokens'] = data['text'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    #data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(data,n)
print(data)

#labeling tweets
def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

predict_x = labelizeTweets(data, 'PREDICT')

#training Word2Vecs
tweet_w2v = Word2Vec(size=n_dim, min_count=2)
xw = []#list(x.words for x in tqdm())
for x in tqdm(predict_x):
    xw.append(x.words)
print("BUILDING")
tweet_w2v.build_vocab(xw)
print("BUILT")
tweet_w2v.train(xw, total_examples = len(xw), epochs = 10)

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1)
matrix = vectorizer.fit_transform([x.words for x in predict_x])
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

vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, predict_x))])

vecs_w2v = scale(vecs_w2v)
print(vecs_w2v)


print("HERE SAFELY")

smo = load_model('sentimentanalysis.model')
print("TESTING")
smo.summary()

res = smo.predict_classes(vecs_w2v)
print("Current Sentiment: ", res)