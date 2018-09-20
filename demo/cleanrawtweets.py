import os
import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def convert(str, startTT):
    return CleanRawTweets.convertToTT(str) - startTT

def getLocation(coor):
    s = coor.find('[') + 1
    e = coor.find(']')
    locString = coor[s:e]
    m = locString.find(',')
    loc = []
    loc.append(float(locString[:m]))
    loc.append(float(locString[m+1:]))
    return loc



class CleanRawTweets(object):

    def procesingRawTweet(config):
        df = pd.DataFrame()
        list = []
        folder = config['file']['input']['raw_tweets']
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                file_path = folder+"\\"+file
                print(file_path)
                sub_df = pd.read_csv(file_path, index_col=None, header=0, encoding='latin-1')

                # remove tweet with coordinates = Null
                sub_df = sub_df.dropna(subset=['coordinates'])

                #remove tweet with land != en
                sub_df[sub_df.lang == "en"]

                # replace text = cleaned tweet
                sub_df['text'] = sub_df.apply(lambda row: CleanRawTweets.remove_breakline(row['text']), axis=1)

                # coordinates
                sub_df['lng'] = sub_df.apply(lambda row: getLocation(row['coordinates'])[0], axis=1)
                sub_df['lat'] = sub_df.apply(lambda row: getLocation(row['coordinates'])[1], axis=1)


                list.append(sub_df)
        df = pd.concat(list)
        print("There are %d items in df" % len(df))
        print("-----------------------------------")

        # timestamp transforming
        startTTString = df.time.values[0]
        print(startTTString)
        startTT = CleanRawTweets.convertToTT(startTTString)
        df['timestamp'] = df.apply(lambda row: convert(row['time'], startTT), axis=1)

        df = df.drop(['id','user_id','lang','time', 'coordinates'], axis = 1)
        df = df.reset_index(drop=True)
        df.index.names = ['id']


        # save cleaned tweets
        print("Saving cleaned tweets")
        outputfile = config['file']['input']['cleaned_tweets']
        df.to_csv(outputfile)
        print("Saving done!!!")

    def remove_breakline(text):
        text = re.sub('\n|\r', ' ', text)
        return text

    @staticmethod
    def tweet_cleaner(text):
        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        try:
            bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            bom_removed = souped
        stripped = re.sub(combined_pat, '', bom_removed)
        stripped = re.sub(www_pat, '', stripped)
        lower_case = stripped.lower()
        neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
        letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
        # During the letters_only process two lines above, it has created unnecessay white spaces,
        # I will tokenize and join together to remove unneccessary white spaces
        words = [x for x in tok.tokenize(letters_only) if len(x) > 1]
        return (" ".join(words)).strip()

    def convertToTT(str):
        return int(time.mktime(time.strptime(str, "%a %b %d %H:%M:%S +0000 %Y")))

