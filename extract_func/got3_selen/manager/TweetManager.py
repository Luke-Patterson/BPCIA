import urllib.request, urllib.parse, urllib.error,urllib.request,urllib.error,urllib.parse,json,re,datetime,sys,http.cookiejar
from .. import models
from pyquery import PyQuery
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import glob
import os
import time
import json

class TweetManager:

	def __init__(self):
		pass

	@staticmethod
	def getTweets(tweetCriteria, receiveBuffer=None, bufferLength=100, proxy=None,halt_count=None):
		refreshCursor = ''

		results = []
		resultsAux = []
		cookieJar = http.cookiejar.CookieJar()

		active = True
		batch_num = 0

		# start selenium driver
		chrome_options = Options()
		chrome_options.add_argument("--headless")
		driver = webdriver.Chrome(executable_path= \
			"C:/Users/lpatterson/AnacondaProjects/chrome_driver/chromedriver.exe")
		actionChains = ActionChains(driver)

		while active:
			time.sleep(1.5)
			if halt_count != None:
				if batch_num > halt_count:
					print('hit batch halt count ' + str(halt_count))
					break
			batch_num +=1
			print('Getting Json batch #' + str(batch_num))
			json = TweetManager.getJsonReponse(tweetCriteria, refreshCursor, cookieJar, proxy,driver = driver)
			if len(json['items_html'].strip()) == 0:
				break

			refreshCursor = json['min_position']
			scrapedTweets = PyQuery(json['items_html'])
			#Remove incomplete tweets withheld by Twitter Guidelines
			scrapedTweets.remove('div.withheld-tweet')
			tweets = scrapedTweets('div.js-stream-tweet')

			if len(tweets) == 0:
				break

			for tweetHTML in tweets:
				tweetPQ = PyQuery(tweetHTML)
				tweet = models.Tweet()

				usernameTweet = tweetPQ("span.username.js-action-profile-name b").text()
				txt = re.sub(r"\s+", " ", tweetPQ("p.js-tweet-text").text().replace('# ', '#').replace('@ ', '@'))
				retweets = int(tweetPQ("span.ProfileTweet-action--retweet span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""))
				favorites = int(tweetPQ("span.ProfileTweet-action--favorite span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""))
				dateSec = int(tweetPQ("small.time span.js-short-timestamp").attr("data-time"))
				id = int(tweetPQ.attr("data-tweet-id"))
				permalink = tweetPQ.attr("data-permalink-path")
				user_id = int(tweetPQ("a.js-user-profile-link").attr("data-user-id"))

				geo = ''
				geoSpan = tweetPQ('span.Tweet-geo')
				if len(geoSpan) > 0:
					geo = geoSpan.attr('title')
				urls = []
				for link in tweetPQ("a"):
					try:
						urls.append((link.attrib["data-expanded-url"]))
					except KeyError:
						pass
				tweet.id = id
				tweet.permalink = 'https://twitter.com' + permalink
				tweet.username = usernameTweet

				tweet.text = txt
				tweet.date = datetime.datetime.fromtimestamp(dateSec)
				tweet.formatted_date = datetime.datetime.fromtimestamp(dateSec).strftime("%a %b %d %X +0000 %Y")
				tweet.retweets = retweets
				tweet.favorites = favorites
				tweet.mentions = " ".join(re.compile('(@ \\w*)').findall(tweet.text))
				tweet.hashtags = " ".join(re.compile('(# \\w*)').findall(tweet.text))
				tweet.geo = geo
				tweet.urls = ",".join(urls)
				tweet.author_id = user_id

				results.append(tweet)
				resultsAux.append(tweet)

				if receiveBuffer and len(resultsAux) >= bufferLength:
					receiveBuffer(resultsAux)
					resultsAux = []

				if tweetCriteria.maxTweets > 0 and len(results) >= tweetCriteria.maxTweets:
					active = False
					break


		if receiveBuffer and len(resultsAux) > 0:
			receiveBuffer(resultsAux)

		# close out selenium webdriver instance
		driver.close()

		return results

	@staticmethod
	def getJsonReponse(tweetCriteria, refreshCursor, cookieJar, proxy,driver):
		url = "https://twitter.com/i/search/timeline?f=tweets&q=%s&src=typd&%smax_position=%s"

		urlGetData = ''
		if hasattr(tweetCriteria, 'username'):
			urlGetData += ' from:' + tweetCriteria.username

		if hasattr(tweetCriteria, 'since'):
			urlGetData += ' since:' + tweetCriteria.since

		if hasattr(tweetCriteria, 'until'):
			urlGetData += ' until:' + tweetCriteria.until

		if hasattr(tweetCriteria, 'querySearch'):
			urlGetData += ' ' + tweetCriteria.querySearch

		if hasattr(tweetCriteria, 'lang'):
			urlLang = 'lang=' + tweetCriteria.lang + '&'
		else:
			urlLang = ''
		url = url % (urllib.parse.quote(urlGetData), urlLang, refreshCursor)
		#print(url)

		headers = [
			('Host', "twitter.com"),
			('User-Agent', "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"),
			('Accept', "application/json, text/javascript, */*; q=0.01"),
			('Accept-Language', "de,en-US;q=0.7,en;q=0.3"),
			('X-Requested-With', "XMLHttpRequest"),
			('Referer', url),
			('Connection', "keep-alive")
		]

		# clear out json files from downloads folder, which selenium will be using
		dl_folder = 'C:/Users/lpatterson/Downloads'
		for fl in glob.glob(dl_folder+'/json*.json'):
			os.remove(fl)

		# go to url in selenium
		count = 0
		while True:
			try:
				driver.get(url)
				break
			except:
				if count<= 20:
					print('Error opening json URL, attempt #' + str(count))
					time.sleep(1)
					count += 1
				else:
					raise('too many attempts to open Twitter JSON URL')
		# keep trying to open json file
		count = 0
		time.sleep(.5)
		while True:
			try:
				with open(dl_folder+'/json.json') as json_file:
					dataJson = json.load(json_file)
				break
			except:
				if count<= 20:
					print('Error opening json, attempt #' + str(count))
					time.sleep(1)
					count += 1
					continue
				else:
					raise('too many attempts to open JSON download')
		return dataJson
