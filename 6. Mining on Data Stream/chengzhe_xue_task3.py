import tweepy
import random

consumer_token = 'tZcmxPJH7szqWTf1A7ITd7PhO'
consumer_secret = 'zIMQvgS1nl4CxvSemgyLmg8YVS8BKJF8LUzCquZaPcQITpkzLo'
access_token = '1121762314586402816-JuINCLc1NAG9EQM2otxDtgq6TA4Raa'
access_token_secret = 'MLAbkdZJXPWvG5o1ngrDSBbzWdAcf4BNMJSphbLEr50AP'

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
dic_hashtags = dict()
list_tweets = list()
count = 1


def print_top_three(x):
    temp_dic = dict()
    for key in x:
        if x[key] not in temp_dic:
            temp_dic[x[key]] = list()
        temp_dic[x[key]].append(key)
    sorted_key = sorted(temp_dic, reverse=True)
    for i in range(3):
        if i >= len(sorted_key):
            break
        else:
            tag_list = sorted(temp_dic[sorted_key[i]])
            for tag in tag_list:
                print(tag, ':', sorted_key[i])


class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        global count
        if len(status.entities['hashtags']) != 0:
            print('\nThe number of tweets with tags from the beginning:', count)
            if count <= 100:
                list_tweets.append(status)
                for dics in status.entities['hashtags']:
                    tag = dics['text']
                    if tag not in dic_hashtags:
                        dic_hashtags[tag] = 0
                    dic_hashtags[tag] += 1
            else:
                keep_prob = 100/count
                ran = random.random()
                if ran < keep_prob:
                    remove_index = random.randint(0, 99)
                    toremove = list_tweets.pop(remove_index)
                    for dics in toremove.entities['hashtags']:
                        tag = dics['text']
                        dic_hashtags[tag] -= 1
                    list_tweets.append(status)
                    for dics in status.entities['hashtags']:
                        tag = dics['text']
                        if tag not in dic_hashtags:
                            dic_hashtags[tag] = 0
                        dic_hashtags[tag] += 1
            count += 1
            print_top_three(dic_hashtags)


myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
myStream.filter(track=['movie', 'music'])
