import re
import csv
import twitter
import time
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import date
from datetime import timedelta
from datetime import datetime
import pandas as pd 

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
								access_token_secret = apikeys[3],
								tweet_mode = 'extended')

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
			tweets_fetched = self.twitter_api.GetSearch(raw_query = keyword)
			print('fetched ' + str(len(tweets_fetched)) + ' tweets for ' + keyword)			
			for status in tweets_fetched:
				self.normal_tweets.append({'text':status.full_text,'id':status.id,'date':status.created_at,'label':None})							
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
		print()
		for tweet in self.normal_tweets:
			print(tweet['text'],tweet['id'])
		print()

	def print_processed_tweets(self):
		for tweet in self.processed_tweets:
			print(tweet)

	def getTweetKeys(self):
		stats = self.twitter_api.GetUserTimeline(screen_name='@Microsoft',count=1)		
		tweet = stats[0].AsDict()
		tweetkeys = []
		for key in tweet.keys():
			tweetkeys.append(key)
		return tweetkeys


	def getWeekTweets(self,company):		
		tdate = date.today()
		week = timedelta(7)
		odate = tdate - week		
		tweetCnt = 0

		self.normal_tweets = []
		query = 'q=to%3A' + company + '%20-filter%3Alinks' + '&since=' + str(odate) + '&count=100'
		self.build_testset_keyword(query)
		tweetCnt = len(self.normal_tweets)

		last_tweet = self.normal_tweets[-1]		
		dt = datetime.strptime(last_tweet['date'],'%a %b %d %H:%M:%S %z %Y').date()		
					
					
		while(dt > odate):			
			query = 'q=to%3A' + company + '%20-filter%3Alinks' + '&since=' + str(odate) + '&max_id=' + str(last_tweet['id']) + '&count=100'
			self.build_testset_keyword(query)							
			if(tweetCnt == len(self.normal_tweets)):
				break
			tweetCnt = len(self.normal_tweets)
			last_tweet = self.normal_tweets[-1]
			dt = datetime.strptime(last_tweet['date'],'%a %b %d %H:%M:%S %z %Y').date()			
			print(dt)
		
		
		self._preprocess_tweets()



	def GetUserTweets(self,usernm,company):
		self.normal_tweets = []

		stats = self.twitter_api.GetUserTimeline(screen_name='@'+usernm,count=200)		
		tweets = [i.AsDict() for i in stats]		
		for tweet in tweets:
			if(company.lower() in tweet['full_text'].lower()):
				self.normal_tweets.append({'text':tweet['full_text'],'id':tweet['id'],'label':None})

		max_tweets = 3240
		iterations = int(max_tweets/200)
		for i in range(iterations):
			stats = self.twitter_api.GetUserTimeline(screen_name=usernm,max_id=tweets[-1]['id'],count=200)
			tweets = [i.AsDict() for i in stats]			
			for tweet in tweets:
				if(company.lower() in tweet['full_text'].lower()):
					self.normal_tweets.append({'text':tweet['full_text'],'id':tweet['id'],'label':None})
		
		print('fetched ' + str(len(self.normal_tweets)) + ' tweets for ' + company + ' from @' + usernm)			


		userdatafile = '.\\data\\twitter\\' + company + '_finance.csv'
		with open(userdatafile,'a') as csvfile:
			lineWriter = csv.writer(csvfile,delimiter=',',quotechar='\"')
			for tweet in self.normal_tweets:
				try:
					lineWriter.writerow([tweet['text'],tweet['id'],tweet['label']])
				except Exception as e:
					print(e)

		self._preprocess_tweets()


	def socialTweets(self,company):
		self.normal_tweets = []
		query = 'q=to%3A' + company + '%20-filter%3Alinks' + '&count=100'

		self.build_testset_keyword(query)

		total_tweets = 2500
		iterations = int(total_tweets/100)

		for i in range(iterations):
			last_tweet = self.normal_tweets[-1]	
			query = 'q=to%3A' + company + '%20-filter%3Alinks' + '&max_id='+ str(last_tweet['id']) + '&count=100'
			self.build_testset_keyword(query)


		userdatafile = '.\\data\\twitter\\' + company + '_peepeepoopoo.csv'
		with open(userdatafile,'w') as csvfile:
			lineWriter = csv.writer(csvfile,delimiter=',',quotechar='\"')
			for tweet in self.normal_tweets:
				try:
					lineWriter.writerow([tweet['text'],tweet['id'],tweet['label']])
				except Exception as e:
					print(e)

		self._preprocess_tweets()

		


if __name__=='__main__':

	companies = ['Apple','Amazon','Facebook','GoldmanSachs','Google','Intel','jpmorgan','Microsoft','Tesla','Samsung','Walmart']
	users = ['CNBC','Benzinga','Stocktwits','BreakoutStocks','bespokeinvest','WSJmarkets','Stephanie_Link','nytimesbusiness','IBDinvestors','WSJdeals']

	api_file = 'APIkeys.txt'

	twitter_time = Da_tweets(api_file)

	twitter_time.getWeekTweets('Amazon')
	print("NORMAL")
	twitter_time.print_tweets()
	print("PROCESSED")
	twitter_time.print_processed_tweets()
	tweets = pd.DataFrame(twitter_time.normal_tweets['text'])
	tweets.to_csv('.\\Sentiment Analysis\\recent_tweets.csv')