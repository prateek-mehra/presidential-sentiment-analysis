#Importing the required libraries
from textblob import Word,TextBlob
import tweepy,sys
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import config


#Helper methods to get the percentage sentiment value, subjectivity and polarity of each tweet
def percentage(part, whole):
    return 100 * float(part)/float(whole)

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

#Returning an overall verdict relating to the analysis of the tweets of a partivular topic
def getAnalysis(score):
    if score == 0:
        return 'Neutral'
    elif score > 0:
        return 'Positive'
    else:
        return 'Negative'
    

#Performing pre-processing of the tweets including lemmatization
def cleanTxt(text,stop_words):
    text = text.replace(r'[^\w\S]','')
    text = re.sub(r'@[A-Za-z0-9]+','',text) #Removing @ mentions
    text = re.sub(r'#','',text) #Removing hashtags
    text = re.sub(r'RT[\s]:+','',text) #Removing RT
    text = re.sub(r'https?:\/\/\S+','',text) #Removing Hyperliink
    text = " ".join(word for word in text.split() if word not in stop_words) #Performing lemmatization
    text = " ".join(Word(word).lemmatize() for word in text.split())
    return text

#Creating a scatter plot of Subjectivity vs Polarity
def plotscatter(df):
    plt.figure(figsize=(8,6))

    for i in range(0,df.shape[0]):
        plt.scatter(df['Polarity'][i],df['Subjectivity'][i],color='Blue')

    plt.title('Sentiment Analysis')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    plt.show()

#Creating a bar depicting the sentiment vs their counts 
def plotbar(df):
    plt.title('Sentiment Analysis')
    plt.xlabel('Sentiments')
    plt.ylabel('Counts')
    df['Analysis'].value_counts().plot(kind = 'bar')
    plt.show()

#Getting the percentage positive tweets relating to a particular topic
def get_pos_percentage(df):
    ptweets = df[df.Analysis == 'Positive']
    ptweets = ptweets['Tweets']

    pos_percentage = round((ptweets.shape[0]/df.shape[0]) * 100, 2)

    print(pos_percentage)

#Getting the percentage negative tweets relating to a particular topic
def get_neg_percentage(df):
    ntweets = df[df.Analysis == 'Negative']
    ntweets = ntweets['Tweets']

    neg_percentage = round((ntweets.shape[0]/df.shape[0]) * 100, 2)

    print(neg_percentage)


#Getting the most positive and negative tweets by sorting them by their polatiry value
def get_neg_tweets(df):
    sortedDF = df.sort_values(by = 'Polarity',ascending = 'False')

    for i in range(0,sortedDF.shape[0]):
        print(sortedDF['Tweets'][i])

def get_pos_tweets(df):
    sortedDF = df.sort_values(by = 'Polarity')

    for i in range(0,sortedDF.shape[0]):
        print(sortedDF['Tweets'][i])

#Identifying the candidate who is the subject of a particular tweet
def identify_subject(tweet, refs):
    flag = 0
    for ref in refs:
        if tweet.find(ref) != -1:
            flag = 1
    return flag

#Sentiment Analysis regarding a particular topic using pandas
def pdAnalysis(posts, searchTerm, noOfSearchTerm):
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns = ['Tweets'])
    df['Tweets'] = df['Tweets'].apply(cleanTxt)
    df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
    df['Polarity'] = df['Tweets'].apply(getPolarity)
    df['Analysis'] = df['Polarity'].apply(getAnalysis)
    pd.set_option('display.max_columns', None)
    #print(df)

    plotscatter(df)
    plotbar(df)
    # get_pos_tweets(df)
    # get_neg_tweets(df)
    # get_pos_percentage(df)
    # get_neg_percentage(df)

#Sentiment Analysis relating to a particular presidential candidate, analyzed by fetching the tweets containing the term: #presidentialdebate .
def prezAnalysis():
    hashtag = "#presidentialdebate"
    query = tweepy.Cursor(api.search,q=hashtag).items(100)
    # c=0
    # for tweet in query:
    #     print(tweet.text.encode('utf8'))
    #     c=c+1
    #     print(c)
    tweets = [{'Tweets':tweet.text,'Timestamp':tweet.created_at} for tweet in query]
    df = pd.DataFrame.from_dict(tweets)
    #nltk.download('stopwords')
    #nltk.download('wordnet')
    stop_words = stopwords.words('english')
    trump_refs = ['DonaldTrump', 'Donald Trump', 'Donald', 'Trump', 'Trump\'s']
    biden_refs = ['JoeBiden', 'Joe', 'Biden', 'Joe Biden', 'Biden\'s']


    #Identifying the subject of a tweet
    df['Trump'] = df['Tweets'].apply(lambda x: identify_subject(x,trump_refs))
    df['Biden'] = df['Tweets'].apply(lambda x: identify_subject(x,biden_refs))

    #Performing the pre-processing of the tweets including lemmatization and storing them in a separate dataframe
    df['Processed Tweets'] = df['Tweets'].apply(lambda x: cleanTxt(x,stop_words))

    #Creating a separate dataframe displaying polarity of a particular tweet
    df['Polarity'] = df['Tweets'].apply(lambda x: TextBlob(x).sentiment[0])
    
    #Creating a separate dataframe displaying the Subjectivity  of a particular tweet
    df['Subjectivity'] = df['Tweets'].apply(lambda x: TextBlob(x).sentiment[1])

    #Calculating the mean, max, min and median polarity with respect to both the candidates
    print(df[df['Trump']==1][['Trump','Polarity','Subjectivity']].groupby('Trump').agg([np.mean,np.max,np.min,np.median]))
    print(df[df['Biden']==1][['Biden','Polarity','Subjectivity']].groupby('Biden').agg([np.mean,np.max,np.min,np.median]))

    #Calculating the moving average polarity with resoect to the presidential election candidates
    biden = df[df['Biden']==1][['Timestamp','Polarity']]
    biden = biden.sort_values(by='Timestamp',ascending=True)
    biden['MA Polarity'] = biden.Polarity.rolling(10,min_periods=3).mean() 

    trump = df[df['Trump']==1][['Timestamp','Polarity']]
    trump = trump.sort_values(by='Timestamp',ascending=True)
    trump['MA Polarity'] = trump.Polarity.rolling(10,min_periods=3).mean() 

    #Visualizaiton of moving average polarity related to the tweets about the candidates
    fig,axes = plt.subplots(2, 1, figsize=(13,10))
    axes[0].plot(biden['Timestamp'],biden['Polarity'])
    axes[0].set_title("\n".join(["Biden Polarity"]))

    axes[1].plot(trump['Timestamp'],trump['Polarity'], color= 'red')
    axes[1].set_title("\n".join(["Trump Polarity"]))


    fig.suptitle("\n".join(["Presidential Debate Analysis"]))
    plt.show()


# A function to perform a basic sentiment analysis of the tweets relating to a particular searchTerm
def BasicAnalysis(posts, searchTerm, noOfSearchTerms):
    
    positive = 0
    negative = 0
    neutral = 0
    polarity = 0

    for tweet in posts:
        #print(tweet.full_text)
        analysis = TextBlob(tweet.full_text)

        polarity += analysis.sentiment.polarity

        if analysis.sentiment.polarity == 0:
            neutral += 1
        elif analysis.sentiment.polarity > 0:
            positive += 1
        elif analysis.sentiment.polarity < 0:
            negative += 1

    #Calculating the percentage positive, nagative and neutral tweets
    positive = percentage(positive, noOfSearchTerms)
    negative = percentage(negative, noOfSearchTerms)
    neutral = percentage(neutral, noOfSearchTerms)
    polarity = percentage(polarity, noOfSearchTerms)

    #Rounding up the calculated sentiments to 2 decimal places
    positive = format(positive, '.2f')
    negative = format(negative, '.2f')
    neutral = format(neutral, '.2f')

    #Displaying the overall polarity
    if(polarity == 0):
        print('Neutral')
    elif (polarity > 0.00):
        print('Positive')
    elif(polarity < 0.00):
        print('Negative')


    #Creating a pie chart depicting the share of positive, negative and neutral tweets
    labels = ['Positive ['+str(positive)+'%]','Negative ['+str(negative)+'%]','Neutral ['+str(neutral)+'%]']
    sizes = [positive,negative,neutral]
    colors = ['green', 'red', 'yellow']
    patches = plt.pie(sizes, colors=colors,  startangle=90)

    #Labelling and showing the plotted chart
    plt.legend(patches, labels, loc="best")
    plt.title('How people are reacting to '+ searchTerm + ' by analysing '+ str(noOfSearchTerms) +' tweets.')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__=="__main__":

    #Performing the auth operations relating to the Tweepy API
   
    auth = tweepy.OAuthHandler(config.consumer_key, config.consumer_secret)

    # try:
    #     redirect_url = auth.get_authorization_url()
    # except tweepy.TweepError:
    #     print('Error! Failed to get request token.')

    # print(redirect_url)
    # verifier = input("Verifier:")

    # try:
    #     auth.get_access_token(verifier)
    # except tweepy.TweepError:
    #     print('Error! Failed to get access token.')

    #auth.set_access_token("bearer_token")

    #print(key,secret)

    auth.set_access_token(config.key ,config.secret)
    # Construct the API instance
    api = tweepy.API(auth)

    #Performing the #presidentialdebate sentiment analysis
    prezAnalysis()

    #Taking the twitter handle of a user and the number of tweets to analyze from the user as input
    searchTerm=input("Enter the handle of the twitter user ")
    noOfSearchTerms = int(input("Enter how many tweets to analyze: "))


    #Performing the basic and intermediate analysis the the user's tweets
    posts = api.user_timeline(screen_name=searchTerm, count = noOfSearchTerms, lang = "en", tweet_mode="extended")
    pdAnalysis(posts, searchTerm, noOfSearchTerms)
    BasicAnalysis(posts, searchTerm, noOfSearchTerms)
