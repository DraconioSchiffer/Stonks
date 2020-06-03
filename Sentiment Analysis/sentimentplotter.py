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




pd.options.mode.chained_assignment = None
tokenizer = TweetTokenizer()
tqdm.pandas(desc="progress-bar")
LabeledSentence = gensim.models.doc2vec.LabeledSentence
n = 2569
n_dim=200

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
tweet_w2v = Word2Vec(size=n_dim, min_count=10)
xw = []#list(x.words for x in tqdm(x_train))
for x in tqdm(x_train):
    xw.append(x.words)
tweet_w2v.build_vocab(xw)
tweet_w2v.train(xw, total_examples = len(xw), epochs = 10)

#defining the chart
output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 2659 word vectors",
    tools="pan,wheel_zoom,box_zoom,reset,hover",
    x_axis_type=None, y_axis_type=None, min_border=1)


word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())[:2000]]

# dimensionality reduction. converting the vectors to 2d vectors
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_w2v = tsne_model.fit_transform(word_vectors)

# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())[:2000]

# plotting. the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"word": "@words"}
show(plot_tfidf)
