""" preProcessTweets contains function used to pre-procees the tweets data"""

import nltk
from nltk.corpus import stopwords       # For Removing Stop words
import re                               # Regular Expressions
from nltk.stem import PorterStemmer     # Stemming
from nltk.tokenize import TweetTokenizer # Tokenizing string
import string


EXAMPLE_TWEET = "RT This has been a #good day. Everyone is safe here https://www.thisisfakewebsite.com"
ENGLISH_STOPWORDS = stopwords.words('english')
STEMMER = PorterStemmer()

"""
    Preprocessing steps in nlp
        Tokenizing the string
        Lowercasing
        Removing stop words and punctuation
        Stemming

"""

def removeStyle(tweet):
    """Remove hyperlinks, twitter marks and styles"""
    
    # Remove Retweet text
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # Remove Links
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # Remove hastags '#'
    tweet = re.sub(r'#', '', tweet)
    return tweet

def tokenizedTweet(tweet):
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                    reduce_len=True)
    token_tweet = tokenizer.tokenize(tweet)
    return token_tweet



def preprocess_text(tweet): 
    tweet =  removeStyle(tweet)
    tokenize_tweet = tokenizedTweet(tweet)
    clean_tweet = [word for word in tokenize_tweet if (word not in ENGLISH_STOPWORDS and word not in string.punctuation)]
    stem_tweet = [STEMMER.stem(word) for word in clean_tweet]

    return stem_tweet


if __name__ == "__main__":
    print(preprocess_text(EXAMPLE_TWEET))
    