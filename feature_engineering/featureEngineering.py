import pandas
from textblob import TextBlob

dataset = pandas.read_csv("data/cleaned_amazon_alexa.csv")

# Length
dataset["length"] = dataset["verified_reviews"].apply(len)
print(dataset["length"])


# Polarity and Subjectivity
def get_polarity(text):
    textblob = TextBlob(str(text.encode("utf-8")))
    pol = textblob.sentiment.polarity
    return pol


def get_subjectivity(text):
    textblob = TextBlob(str(text.encode("utf-8")))
    pol = textblob.sentiment.subjectivity
    return pol


dataset["Polarity"] = dataset["verified_reviews"].apply(get_polarity)
dataset["Subjectivity"] = dataset["verified_reviews"].apply(get_subjectivity)

print(dataset[['length', 'Polarity', 'Subjectivity']])

# Words , Characters and punctuation
dataset["charsCount"] = dataset["verified_reviews"].apply(len)
dataset["wordCount"] = dataset["verified_reviews"].apply(lambda x: len(x.split()))
dataset["word_density"] = dataset["charsCount"] / (dataset["wordCount"] + 1)

print(dataset[['charsCount', 'wordCount', 'word_density']])


# Nouns , Verbs , pronoun , Adverbs , Adj
def get_nouns(text):
    list_of_nouns = []
    textblob = TextBlob(str(text.encode("utf-8")))
    tags = textblob.tags
    for i in tags:
        if i[1] == "NN":
            list_of_nouns.append(i[0])
    return list_of_nouns


def get_verbs(text):
    list_of_verbs = []
    textblob = TextBlob(str(text.encode("utf-8")))
    tags = textblob.tags
    for i in tags:
        if "VB" in i[1]:
            list_of_verbs.append(i[0])
    return list_of_verbs


def get_pronouns(text):
    list_of_pronouns = []
    textblob = TextBlob(str(text.encode("utf-8")))
    tags = textblob.tags
    for i in tags:
        if i[1] == "WP":
            list_of_pronouns.append(i[0])
    return list_of_pronouns


def get_adverbs(text):
    list_of_adverbs = []
    textblob = TextBlob(str(text.encode("utf-8")))
    tags = textblob.tags
    for i in tags:
        if "RB" in i[1]:
            list_of_adverbs.append(i[0])
    return list_of_adverbs


def get_adj(text):
    list_of_adj = []
    textblob = TextBlob(str(text.encode("utf-8")))
    tags = textblob.tags
    for i in tags:
        if "JJ" in i[1]:
            list_of_adj.append(i[0])
    return list_of_adj


dataset["nouns"] = dataset["verified_reviews"].apply(lambda x: get_nouns(x))
dataset["verbs"] = dataset["verified_reviews"].apply(lambda x: get_verbs(x))
dataset["pronouns"] = dataset["verified_reviews"].apply(lambda x: get_pronouns(x))
dataset["adverbs"] = dataset["verified_reviews"].apply(lambda x: get_adverbs(x))
dataset["adj"] = dataset["verified_reviews"].apply(lambda x: get_adj(x))

print(dataset[["verified_reviews", "nouns", "verbs", "pronouns", "adverbs", "adj"]])
