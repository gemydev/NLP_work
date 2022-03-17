# In case you have a problem reading your dataset , check its encoding

import chardet
import pandas

with open("McDonalds-Yelp-Sentiment-DFE.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)
dataset = pandas.read_csv("McDonalds-Yelp-Sentiment-DFE.csv", encoding="Windows-1252")
