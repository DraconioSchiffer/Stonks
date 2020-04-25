import re
import csv
import twitter
import time
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Da_tweets:
	
	normal_tweets = []
	processed_tweets = []
	twitter_api = None	

	def __init__(self,api_file):
		self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
		self._initialize_keys(api_file)

	def _initialize_keys(self,api_file):		
		apikeys_file = open(api_file,'r')
		reader = csv.reader(apikeys_file,delimiter=':')

		apikeys = []
		for row in reader:
			apikeys.append(row[1])
		
		self.twitter_api = twitter.Api(consumer_key = apikeys[0],
								consumer_secret = apikeys[1],
								access_token_key = apikeys[2],
								access_token_secret = apikeys[3])

	def test_api(self):
		print(self.twitter_api.VerifyCredentials())


	def build_trainingset(self,corpusfile,tweetdatafile):
		corpus = []
		with open(corpusfile,'r') as csvfile:
			lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
			for row in lineReader:
				corpus.append({"tweet_id":row[2],"label":row[1],"topic":row[0]})					

		rate_limit = 180
		sleep_time = 900/180
		trainingDataset = []

		for tweet in corpus:
			try:
				status = self.twitter_api.GetStatus(tweet["tweet_id"])
				print("Tweet fetched " + status.text)
				tweet["text"] = status.text
				trainingDataset.append(tweet)
				time.sleep(sleep_time)
			except:
				continue

		with open(tweetdatafile,'w') as csvfile:
			lineWriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
			for tweet in trainingDataset:
				try:
					lineWriter.writerow([tweet["tweet_id"],tweet["text"],tweet["label"],tweet["topic"]])
				except Exception as e:
					print(e)
		


	def build_testset_keyword(self,keyword):
		try:			
			tweets_fetched = self.twitter_api.GetSearch(raw_query = keyword ,count = 10)
			print('fetched ' + str(len(tweets_fetched)) + ' tweets for ' + keyword)			
			for status in tweets_fetched:
				self.normal_tweets.append({'text':status.text, 'label':None})
			self._preprocess_tweets()
		except:
			print('Some unfortunate event happened RIP maybe internet issues or stuff')
			return None	


	def _preprocess_tweets(self):
		for tweet in self.normal_tweets:
			temp_tweet = tweet['text'].lower()
			temp_tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', temp_tweet) 
			temp_tweet = re.sub('@[^\s]+', 'AT_USER', temp_tweet)
			temp_tweet = re.sub(r'#([^\s]+)', r'\1', temp_tweet) 
			temp_tweet = word_tokenize(temp_tweet) 
			self.processed_tweets.append(([word for word in temp_tweet if word not in self._stopwords],tweet['label']))


	def print_tweets(self):
		for tweet in self.normal_tweets:
			print(tweet['text'])

	def print_processed_tweets(self):
		for tweet in self.processed_tweets:
			print(tweet)




if __name__=='__main__':

	companies = ['Apple','Amazon','Facebook','Google','Intel','Microsoft','Samsung','Walmart']

	api_file = '.\\APIkeys.txt'
	corpusfile = '.\\data\\corpus.csv'
	tweetfile = '.\\data\\tweetdata.csv'
	twitter_time = Da_tweets(api_file)
	#twitter_time.test_api()
	#twitter_time.build_trainingset(corpusfile,tweetfile)	

	for company in companies:	
		query = 'q=' + company + '%20%23' + company + '%20%40' + company + '%20lang%3Aen%20-filter%3Alinks%20-filter%3Areplies&'		
		twitter_time.build_testset_keyword(query)		
		comptweetsfile = '.\\data\\' + company + '_tweets.csv'		
		with open(comptweetsfile,'w') as csvfile:
			lineWriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
			for tweet in twitter_time.normal_tweets:
				try:
					lineWriter.writerow([tweet["text"],tweet["label"]])
				except Exception as e:
					print(e)
		
	#twitter_time.build_testset_keyword('Microsoft')	
	#twitter_time.print_processed_tweets()
	 